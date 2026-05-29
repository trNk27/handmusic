"""
Hand Shape -> Sound Recognizer (MusicMaker)

Watches the webcam, recognizes the shape of a single hand held up to the camera,
and plays a sound matching the detected shape.

Gestures (hybrid set):
  - Named gestures take priority: fist, open palm, peace, thumbs up, pointing, OK
  - Otherwise falls back to a finger count: count-0 .. count-5

Sound (both modes supported):
  - By default each gesture plays a synthesized musical note.
  - If a matching file exists in sounds/<gesture>.wav it is played instead.

Controls:
  - Q or Esc : quit

Run:
  py hand_music.py
"""

import math
import os
import time

import cv2
import numpy as np
import pygame
import mediapipe as mp

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Camera source. Either:
#   - an integer device index for a local webcam (e.g. 0), or
#   - a stream URL string for an IP / phone camera.
# This is set to the IP Webcam (Android app) MJPEG stream on the phone.
# To use the built-in laptop webcam instead, set CAMERA_SOURCE = 0.
CAMERA_SOURCE = "http://192.168.0.227:8080/video"
SAMPLE_RATE = 44100         # audio sample rate (Hz)
NOTE_DURATION = 0.45        # length of a synthesized note (seconds)
GESTURE_COOLDOWN = 0.6      # min seconds before the same gesture re-triggers
STABLE_FRAMES = 2           # frames a gesture must persist before it counts
SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# Each gesture -> musical note frequency (Hz). Used for the synthesized fallback.
# C major scale across roughly one octave.
NOTE_FREQUENCIES = {
    "fist":      261.63,   # C4
    "thumbs_up": 293.66,   # D4
    "pointing":  329.63,   # E4
    "peace":     349.23,   # F4
    "ok":        392.00,   # G4
    "open_palm": 523.25,   # C5
    "count-0":   261.63,   # C4
    "count-1":   293.66,   # D4
    "count-2":   329.63,   # E4
    "count-3":   392.00,   # G4
    "count-4":   440.00,   # A4
    "count-5":   523.25,   # C5
}

# Human-friendly labels for the on-screen overlay.
GESTURE_LABELS = {
    "fist":      "Fist",
    "thumbs_up": "Thumbs Up",
    "pointing":  "Pointing",
    "peace":     "Peace",
    "ok":        "OK Sign",
    "open_palm": "Open Palm",
    "count-0":   "0 Fingers",
    "count-1":   "1 Finger",
    "count-2":   "2 Fingers",
    "count-3":   "3 Fingers",
    "count-4":   "4 Fingers",
    "count-5":   "5 Fingers",
}

# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
class SoundBank:
    """Lazily produces and caches a pygame Sound per gesture.

    Uses sounds/<gesture>.wav when present, otherwise synthesizes a note.
    """

    def __init__(self):
        # 16-bit stereo mixer so synthesized buffers and .wav files both work.
        pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=2)
        pygame.mixer.init()
        self._cache = {}

    def _synthesize(self, freq):
        n_samples = int(SAMPLE_RATE * NOTE_DURATION)
        t = np.linspace(0.0, NOTE_DURATION, n_samples, endpoint=False)
        # Sine tone plus a soft second harmonic for a less plain timbre.
        wave = 0.6 * np.sin(2 * math.pi * freq * t)
        wave += 0.2 * np.sin(2 * math.pi * freq * 2 * t)

        # Attack/decay envelope to avoid clicks at start/end.
        envelope = np.ones(n_samples)
        attack = int(0.01 * SAMPLE_RATE)
        release = int(0.15 * SAMPLE_RATE)
        envelope[:attack] = np.linspace(0.0, 1.0, attack)
        envelope[-release:] = np.linspace(1.0, 0.0, release)
        wave *= envelope

        audio = np.int16(wave * 32767)
        stereo = np.column_stack((audio, audio))  # duplicate to 2 channels
        return pygame.sndarray.make_sound(np.ascontiguousarray(stereo))

    def get(self, gesture):
        if gesture in self._cache:
            return self._cache[gesture]

        wav_path = os.path.join(SOUNDS_DIR, gesture + ".wav")
        if os.path.isfile(wav_path):
            try:
                sound = pygame.mixer.Sound(wav_path)
            except pygame.error as exc:
                print(f"[warn] could not load {wav_path}: {exc}; using a note instead")
                sound = self._synthesize(NOTE_FREQUENCIES.get(gesture, 440.0))
        else:
            sound = self._synthesize(NOTE_FREQUENCIES.get(gesture, 440.0))

        self._cache[gesture] = sound
        return sound

    def play(self, gesture):
        self.get(gesture).play()


# --------------------------------------------------------------------------- #
# Gesture detection
# --------------------------------------------------------------------------- #
def fingers_up(landmarks, handedness_label):
    """Return [thumb, index, middle, ring, pinky] booleans for raised fingers.

    Image coordinates: y grows downward, so a raised finger has tip.y < pip.y.
    """
    lm = landmarks.landmark

    # Thumb: extension is sideways, so compare x. Direction depends on which hand.
    # The frame is mirrored, so MediaPipe's "Right" label is the user's left hand.
    if handedness_label == "Right":
        thumb = lm[THUMB_TIP].x < lm[THUMB_IP].x
    else:
        thumb = lm[THUMB_TIP].x > lm[THUMB_IP].x

    index = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    middle = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y
    ring = lm[RING_TIP].y < lm[RING_PIP].y
    pinky = lm[PINKY_TIP].y < lm[PINKY_PIP].y

    return [bool(thumb), bool(index), bool(middle), bool(ring), bool(pinky)]


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def classify(landmarks, handedness_label):
    """Classify the hand into a named gesture, else a finger-count gesture."""
    up = fingers_up(landmarks, handedness_label)
    thumb, index, middle, ring, pinky = up
    lm = landmarks.landmark

    # OK sign: thumb tip touching index tip while the other fingers are extended.
    # Use hand size (wrist -> middle PIP) to scale the "touching" threshold.
    hand_scale = _distance(lm[WRIST], lm[MIDDLE_PIP]) + 1e-6
    pinch = _distance(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_scale
    if pinch < 0.35 and middle and ring and pinky:
        return "ok"

    # Named gestures by exact finger pattern.
    if up == [0, 0, 0, 0, 0]:
        return "fist"
    if up == [1, 1, 1, 1, 1]:
        return "open_palm"
    if up == [0, 1, 1, 0, 0]:
        return "peace"
    if up == [1, 0, 0, 0, 0]:
        return "thumbs_up"
    if up == [0, 1, 0, 0, 0]:
        return "pointing"

    # Fall back to a finger count.
    return f"count-{sum(up)}"


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
def draw_overlay(frame, gesture, playing):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    if gesture:
        label = GESTURE_LABELS.get(gesture, gesture)
        note = NOTE_FREQUENCIES.get(gesture)
        text = f"{label}"
        if note:
            text += f"   ~{note:.0f} Hz"
        color = (0, 255, 120) if playing else (220, 220, 220)
    else:
        text = "No hand detected"
        color = (0, 0, 255)

    cv2.putText(frame, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, "Press Q to quit", (w - 230, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    bank = SoundBank()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera source {CAMERA_SOURCE!r}. "
            "For a phone IP camera, make sure the app is running and the phone "
            "is on the same network. For a local webcam, set CAMERA_SOURCE = 0 "
            "at the top of hand_music.py."
        )

    # Keep latency low on network streams by not letting frames queue up.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    last_gesture = None         # last gesture that triggered a sound
    last_play_time = 0.0
    candidate = None            # gesture awaiting stability confirmation
    candidate_count = 0

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        print("Running. Show a hand to the camera. Press Q to quit.")
        read_failures = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                # Network streams occasionally drop a frame; tolerate a few
                # transient failures before giving up.
                read_failures += 1
                if read_failures > 30:
                    print("[error] lost the camera stream; exiting")
                    break
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            read_failures = 0

            frame = cv2.flip(frame, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = None
            just_played = False

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label

                mp_draw.draw_landmarks(
                    frame, landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                gesture = classify(landmarks, handedness)

                # Require the gesture to persist a couple of frames before acting.
                if gesture == candidate:
                    candidate_count += 1
                else:
                    candidate = gesture
                    candidate_count = 1

                if candidate_count >= STABLE_FRAMES:
                    now = time.time()
                    changed = gesture != last_gesture
                    cooled = (now - last_play_time) >= GESTURE_COOLDOWN
                    if changed or cooled:
                        bank.play(gesture)
                        last_gesture = gesture
                        last_play_time = now
                        just_played = True
                        print(f"-> {GESTURE_LABELS.get(gesture, gesture)}")
            else:
                # No hand: reset so re-showing the same gesture plays again.
                candidate = None
                candidate_count = 0
                last_gesture = None

            draw_overlay(frame, gesture, just_played)
            cv2.imshow("MusicMaker - Hand Shape Sounds", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # q or Esc
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()

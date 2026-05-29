// MusicMaker — browser version
// Recognizes hand shapes from the camera (MediaPipe HandLandmarker) and plays a
// sound for each. Runs 100% client-side; suitable for GitHub Pages.

import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

// --------------------------------------------------------------------------- //
// Configuration
// --------------------------------------------------------------------------- //
const NOTE_DURATION = 0.45; // seconds (synthesized note length)
const GESTURE_COOLDOWN = 0.6; // seconds before the same gesture re-triggers
const STABLE_FRAMES = 2; // frames a gesture must persist before it counts
const SOUNDS_DIR = "./sounds"; // optional <gesture>.mp3 files live here

// Two-hand theremin: maps the normalized distance between the hands to a pitch.
// Hands close together -> low note; far apart -> high note (continuous glide).
const THEREMIN = { dMin: 0.1, dMax: 0.7, fMin: 150, fMax: 900, volume: 0.22 };

// Gesture -> musical note frequency (Hz), C major scale. Synthesized fallback.
const NOTE_FREQUENCIES = {
  fist: 261.63, // C4
  thumbs_up: 293.66, // D4
  pointing: 329.63, // E4
  peace: 349.23, // F4
  ok: 392.0, // G4
  open_palm: 523.25, // C5
  "count-0": 261.63,
  "count-1": 293.66,
  "count-2": 329.63,
  "count-3": 392.0,
  "count-4": 440.0,
  "count-5": 523.25,
};

const GESTURE_LABELS = {
  fist: "Fist",
  thumbs_up: "Thumbs Up",
  pointing: "Pointing",
  peace: "Peace",
  ok: "OK Sign",
  open_palm: "Open Palm",
  "count-0": "0 Fingers",
  "count-1": "1 Finger",
  "count-2": "2 Fingers",
  "count-3": "3 Fingers",
  "count-4": "4 Fingers",
  "count-5": "5 Fingers",
};

// MediaPipe hand landmark indices (same as the Python version).
const WRIST = 0;
const THUMB_TIP = 4, THUMB_IP = 3;
const INDEX_TIP = 8, INDEX_PIP = 6;
const MIDDLE_TIP = 12, MIDDLE_PIP = 10;
const RING_TIP = 16, RING_PIP = 14;
const PINKY_TIP = 20, PINKY_PIP = 18;
const PALM = 9; // middle-finger MCP, used as a stable palm center

// --------------------------------------------------------------------------- //
// Audio (Web Audio API): synthesized notes + optional file overrides
// --------------------------------------------------------------------------- //
let audioCtx = null;
const fileBuffers = {}; // gesture -> AudioBuffer (when a file was found)

function initAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") audioCtx.resume();
}

// Try to preload sounds/<gesture>.mp3 for every gesture. Missing files are
// simply ignored, leaving the synthesized note as the fallback ("both" mode).
async function preloadSoundFiles() {
  const names = Object.keys(NOTE_FREQUENCIES);
  await Promise.all(
    names.map(async (name) => {
      try {
        const res = await fetch(`${SOUNDS_DIR}/${name}.mp3`, { cache: "force-cache" });
        if (!res.ok) return;
        const buf = await res.arrayBuffer();
        fileBuffers[name] = await audioCtx.decodeAudioData(buf);
      } catch {
        /* no file for this gesture — use synthesized note */
      }
    })
  );
}

function playFile(buffer) {
  const src = audioCtx.createBufferSource();
  src.buffer = buffer;
  src.connect(audioCtx.destination);
  src.start();
}

function playSynth(freq) {
  const now = audioCtx.currentTime;
  const gain = audioCtx.createGain();
  gain.connect(audioCtx.destination);
  // Attack/release envelope to avoid clicks.
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.linearRampToValueAtTime(0.35, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + NOTE_DURATION);

  // Fundamental + soft second harmonic for a less plain timbre.
  const osc1 = audioCtx.createOscillator();
  osc1.type = "sine";
  osc1.frequency.value = freq;
  const osc2 = audioCtx.createOscillator();
  osc2.type = "sine";
  osc2.frequency.value = freq * 2;
  const h2 = audioCtx.createGain();
  h2.gain.value = 0.3;

  osc1.connect(gain);
  osc2.connect(h2).connect(gain);
  osc1.start(now);
  osc2.start(now);
  osc1.stop(now + NOTE_DURATION + 0.02);
  osc2.stop(now + NOTE_DURATION + 0.02);
}

function playGesture(gesture) {
  if (fileBuffers[gesture]) {
    playFile(fileBuffers[gesture]);
  } else {
    playSynth(NOTE_FREQUENCIES[gesture] ?? 440);
  }
}

// Continuous "theremin" voice: one long-running oscillator whose pitch and
// volume we steer in real time. Created lazily on first use, then left running
// (silenced via its gain) so there are no start/stop clicks.
let theremin = null;

function thereminUpdate(freq) {
  if (!theremin) {
    const osc = audioCtx.createOscillator();
    osc.type = "sine";
    osc.frequency.value = freq;
    const gain = audioCtx.createGain();
    gain.gain.value = 0;
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    theremin = { osc, gain };
  }
  const now = audioCtx.currentTime;
  // setTargetAtTime gives a smooth glide rather than stepping per frame.
  theremin.osc.frequency.setTargetAtTime(freq, now, 0.02);
  theremin.gain.gain.setTargetAtTime(THEREMIN.volume, now, 0.02);
}

function thereminStop() {
  if (!theremin) return;
  theremin.gain.gain.setTargetAtTime(0.0001, audioCtx.currentTime, 0.05);
}

function distanceToFreq(d) {
  const t = Math.min(1, Math.max(0, (d - THEREMIN.dMin) / (THEREMIN.dMax - THEREMIN.dMin)));
  return THEREMIN.fMin + t * (THEREMIN.fMax - THEREMIN.fMin);
}

// --------------------------------------------------------------------------- //
// Gesture detection (ported from the Python classifier)
// --------------------------------------------------------------------------- //
function fingersUp(lm, handedness) {
  // Image coords: y grows downward, so a raised finger has tip.y < pip.y.
  // The view is mirrored, so MediaPipe's "Right" label is the user's left hand.
  let thumb;
  if (handedness === "Right") thumb = lm[THUMB_TIP].x < lm[THUMB_IP].x;
  else thumb = lm[THUMB_TIP].x > lm[THUMB_IP].x;

  const index = lm[INDEX_TIP].y < lm[INDEX_PIP].y;
  const middle = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y;
  const ring = lm[RING_TIP].y < lm[RING_PIP].y;
  const pinky = lm[PINKY_TIP].y < lm[PINKY_PIP].y;
  return [thumb, index, middle, ring, pinky].map((b) => (b ? 1 : 0));
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function arrEq(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

function classify(lm, handedness) {
  const up = fingersUp(lm, handedness);
  const [, , middle, ring, pinky] = up;

  // OK sign: thumb tip touching index tip with other fingers extended.
  const handScale = dist(lm[WRIST], lm[MIDDLE_PIP]) + 1e-6;
  const pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / handScale;
  if (pinch < 0.35 && middle && ring && pinky) return "ok";

  if (arrEq(up, [0, 0, 0, 0, 0])) return "fist";
  if (arrEq(up, [1, 1, 1, 1, 1])) return "open_palm";
  if (arrEq(up, [0, 1, 1, 0, 0])) return "peace";
  if (arrEq(up, [1, 0, 0, 0, 0])) return "thumbs_up";
  if (arrEq(up, [0, 1, 0, 0, 0])) return "pointing";

  const count = up.reduce((s, v) => s + v, 0);
  return `count-${count}`;
}

// --------------------------------------------------------------------------- //
// Main
// --------------------------------------------------------------------------- //
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("start");
const gestureEl = document.getElementById("gesture");
const noteEl = document.getElementById("note");

let handLandmarker = null;
let drawingUtils = null;
let running = false;

// Debounce state
let lastGesture = null;
let lastPlayTime = 0;
let candidate = null;
let candidateCount = 0;
let lastVideoTime = -1;
let lastOverlay = { title: "No hand", sub: "" };

function drawHand(lm) {
  drawingUtils.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, {
    color: "#5b8cff",
    lineWidth: 4,
  });
  drawingUtils.drawLandmarks(lm, { color: "#6ee7a8", radius: 4 });
}

// Dashed line between the two palm centers (normalized coords -> canvas pixels).
function drawHandLink(a, b) {
  ctx.save();
  ctx.strokeStyle = "#ffd166";
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 8]);
  ctx.beginPath();
  ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
  ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
  ctx.stroke();
  ctx.restore();
}

async function createLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

function renderLoop() {
  if (!running) return;

  // Draw the mirrored camera frame (selfie view), matching the Python flip.
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(canvas, performance.now());
    const hands = result.landmarks || [];

    if (hands.length >= 2) {
      // --- Two hands: continuous theremin driven by their distance ---
      drawHand(hands[0]);
      drawHand(hands[1]);

      const a = hands[0][PALM];
      const b = hands[1][PALM];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      const freq = distanceToFreq(d);

      drawHandLink(a, b);
      thereminUpdate(freq);

      // Reset single-hand state so a gesture re-triggers when one hand returns.
      candidate = null;
      candidateCount = 0;
      lastGesture = null;

      lastOverlay = { title: "🎻 Theremin", sub: `${Math.round(freq)} Hz` };
    } else if (hands.length === 1) {
      // --- One hand: discrete gesture sounds (theremin off) ---
      thereminStop();
      const lm = hands[0];
      // Handedness flips because we detect on the mirrored canvas — matches Python.
      const rawLabel = result.handednesses?.[0]?.[0]?.categoryName ?? "Right";
      const handedness = rawLabel === "Right" ? "Left" : "Right";

      drawHand(lm);
      const gesture = classify(lm, handedness);

      if (gesture === candidate) candidateCount++;
      else {
        candidate = gesture;
        candidateCount = 1;
      }

      if (candidateCount >= STABLE_FRAMES) {
        const now = performance.now() / 1000;
        const changed = gesture !== lastGesture;
        const cooled = now - lastPlayTime >= GESTURE_COOLDOWN;
        if (changed || cooled) {
          playGesture(gesture);
          lastGesture = gesture;
          lastPlayTime = now;
        }
      }

      const f = NOTE_FREQUENCIES[gesture];
      const sub = fileBuffers[gesture] ? "file" : `${Math.round(f)} Hz`;
      lastOverlay = { title: GESTURE_LABELS[gesture] ?? gesture, sub };
    } else {
      // --- No hands ---
      thereminStop();
      candidate = null;
      candidateCount = 0;
      lastGesture = null;
      lastOverlay = { title: "No hand", sub: "" };
    }
  }

  gestureEl.textContent = lastOverlay.title;
  noteEl.textContent = lastOverlay.sub;

  requestAnimationFrame(renderLoop);
}

startBtn.addEventListener("click", async () => {
  startBtn.textContent = "Loading…";
  startBtn.style.pointerEvents = "none";
  try {
    initAudio();
    await Promise.all([createLandmarker(), startCamera()]);
    await preloadSoundFiles();
    drawingUtils = new DrawingUtils(ctx);
    running = true;
    startBtn.classList.add("hidden");
    renderLoop();
  } catch (err) {
    console.error(err);
    startBtn.style.pointerEvents = "auto";
    startBtn.innerHTML = `<span class="big">⚠ Could not start</span><span>${err.message}</span>`;
  }
});

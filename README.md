# MusicMaker — Hand Shape Sound Recognizer

Hold a hand up to your webcam and the program recognizes its shape and plays a
sound. It uses [MediaPipe](https://developers.google.com/mediapipe) for hand
landmark detection, OpenCV for the live camera window, and pygame for audio.

## Features

- **Hybrid gesture recognition** — named gestures take priority, with a
  finger-count fallback:
  - Named: fist, open palm, peace ✌, thumbs up 👍, pointing 👆, OK 👌
  - Counting: 0, 1, 2, 3, 4, 5 fingers
- **Two sound modes (automatic)** — plays a synthesized musical note by default,
  or your own `sounds/<gesture>.wav` file if one exists.
- **Live webcam window** with the hand skeleton drawn and the detected gesture +
  note shown on screen.
- Debounced playback so each sound triggers once per gesture change instead of
  every frame.

## Setup

Requires Python 3.10 and a webcam. A virtual environment is recommended (and
required if your global Python has an older TensorFlow installed, which conflicts
with MediaPipe):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

(First run downloads MediaPipe's hand model automatically — one time.)

## Run

```powershell
.\.venv\Scripts\python.exe hand_music.py
```

Show a hand to the camera. Press **Q** (or **Esc**) to quit.

## Camera source

The camera is selected by `CAMERA_SOURCE` near the top of
[hand_music.py](hand_music.py). It accepts either:

- an **integer** for a local webcam (e.g. `0`), or
- a **stream URL** for a phone / IP camera.

It is currently set to a phone running the **IP Webcam** Android app:

```python
CAMERA_SOURCE = "http://192.168.0.227:8080/video"
```

To use the IP Webcam phone camera:

1. Install **IP Webcam** on the Android phone and tap *Start server*.
2. Make sure the phone and this PC are on the **same Wi-Fi network**.
3. Use the `/video` MJPEG endpoint of the address the app shows
   (`http://<phone-ip>:8080/video`). Update the IP in `CAMERA_SOURCE` if your
   phone's address differs.

To switch back to the built-in laptop webcam, set `CAMERA_SOURCE = 0`.

## Gesture → sound map

| Gesture     | Default note |
|-------------|--------------|
| Fist        | C4           |
| Thumbs up   | D4           |
| Pointing    | E4           |
| Peace       | F4           |
| OK sign     | G4           |
| Open palm   | C5           |
| 0–5 fingers | C4 … C5      |

To use your own audio, drop WAV files into [sounds/](sounds/) — see
[sounds/README.md](sounds/README.md) for the naming convention.

## Troubleshooting

- **Wrong / black camera:** change `CAMERA_INDEX` near the top of
  [hand_music.py](hand_music.py) (try `1`, `2`, …).
- **No hand detected:** improve lighting and keep the whole hand in frame.
- **No sound:** check system volume / output device; the title-bar overlay still
  shows the detected gesture even if audio is muted.

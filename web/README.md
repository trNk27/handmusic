# MusicMaker — Browser version

A pure client-side port of the Python hand-shape sound recognizer. It uses
[MediaPipe Tasks Vision `HandLandmarker`](https://ai.google.dev/edge/mediapipe)
(WASM, loaded from a CDN), `getUserMedia` for the camera, and the Web Audio API
for sound. No server, no build step — just static files.

## Run locally

Browsers require a secure context for the camera. `http://localhost` counts as
secure, so a tiny local server works:

```powershell
# from the web/ folder
py -m http.server 8000
# then open http://localhost:8000
```

Click **Start camera**, allow access, and show a hand. (The first load downloads
the MediaPipe model + WASM, a few MB.)

## Host on GitHub Pages

1. Push this repo to GitHub.
2. Repo **Settings → Pages → Build and deployment → Source: Deploy from a branch**,
   pick your branch and `/ (root)` or set the folder appropriately.
3. Open the published URL. Because Pages is HTTPS, the camera works — including
   **on your phone**: open the URL on the phone and it uses the phone's camera
   directly (no IP-camera app needed).

> Note: a GitHub Pages (HTTPS) site cannot pull from a phone's `http://…:8080`
> IP-Webcam stream (mixed-content + CORS). Running the page *on the phone* is the
> intended way to use the phone camera here.

## Gestures & sounds

Same hybrid set as the Python version: fist, open palm, peace, thumbs up,
pointing, OK — with a finger-count fallback (0–5). Each plays a synthesized note
by default, or a matching file from [sounds/](sounds/) if present
(see [sounds/README.md](sounds/README.md)).

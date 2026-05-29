# Custom sounds (optional)

Drop `.wav` files here to override the synthesized note for any gesture. The file
name must match the gesture key exactly, with a `.wav` extension.

Named gestures:

| File              | Gesture     |
|-------------------|-------------|
| `fist.wav`        | Fist        |
| `open_palm.wav`   | Open palm   |
| `peace.wav`       | Peace sign  |
| `thumbs_up.wav`   | Thumbs up   |
| `pointing.wav`    | Pointing    |
| `ok.wav`          | OK sign     |

Finger-count fallback:

| File             | Gesture    |
|------------------|------------|
| `count-0.wav`    | 0 fingers  |
| `count-1.wav`    | 1 finger   |
| `count-2.wav`    | 2 fingers  |
| `count-3.wav`    | 3 fingers  |
| `count-4.wav`    | 4 fingers  |
| `count-5.wav`    | 5 fingers  |

If no matching file is present, the program synthesizes a musical note instead.
Only `.wav` is loaded directly; convert other formats to WAV first.

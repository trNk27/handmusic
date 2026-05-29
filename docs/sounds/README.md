# Custom sounds (optional, browser version)

Drop `.mp3` files here to override the synthesized note for any gesture. The file
name must match the gesture key exactly, with a `.mp3` extension. If no matching
file is present, the app synthesizes a musical note instead.

Named gestures: `fist.mp3`, `open_palm.mp3`, `peace.mp3`, `thumbs_up.mp3`,
`pointing.mp3`, `ok.mp3`

Finger-count fallback: `count-0.mp3` … `count-5.mp3`

The browser decodes these with the Web Audio API; `.mp3`, `.wav`, and `.ogg`
generally work, but `.mp3` is the safest cross-browser choice.

# ✋ Air Painter

Draw in the air with your index finger. Your webcam tracks your hand in real time, and a few gestures let you change colors, erase and save your art. No mouse, no touchscreen, no GPU.

![Python](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-0097A7?logo=google&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- ![Air Painter demo](assets/demo.png) -->

## How it works

1. OpenCV reads frames from your webcam and mirrors them.
2. Google's MediaPipe Hands finds 21 landmarks on your hand in each frame.
3. Simple rules check which fingers are up (is a fingertip higher than its middle joint?) and map each gesture to an action.
4. The index fingertip is the pen. Its path is drawn on a separate canvas that is blended over the video.

## Gestures

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Draw |
| ✌️ Index + middle over a color box | Pick that color |
| 🖐️ Open palm, wave around | Erase |
| ✊ Fist | Clear everything |
| Press `s` | Save the drawing (white background) |
| Press `p` | Save a screenshot of the full scene |
| Press `q` | Quit |

Saved files go into a `drawings/` folder next to the script.

## Try it yourself

You need Python 3.10 to 3.12 (MediaPipe doesn't install on newer versions yet) and a webcam.

```bash
git clone https://github.com/PearlM-457/Computer_Vision_Projects.git
cd Computer_Vision_Projects/01-hand-tracking

python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell)
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python hand_track.py
```

Click on the video window before pressing keys.

## Troubleshooting

- **Black window or wrong camera:** change `CAMERA_INDEX` at the top of `hand_track.py` (try 0, then 1).
- **`module 'mediapipe' has no attribute 'solutions'`:** you have a newer MediaPipe. Reinstall with `pip install -r requirements.txt`.
- **Hand isn't detected well:** use good lighting and keep your hand 30 to 80 cm from the camera.
- **Camera light on but no window:** close other apps using the webcam (Zoom, browser tabs).

## Ideas for improvement

- Hold-to-confirm for the fist, so it can't clear your drawing by accident
- Adjustable brush size using a pinch gesture
- Train a small classifier on recorded landmarks instead of hardcoded finger rules
- Undo last stroke

## Built with

[OpenCV](https://opencv.org/), [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), NumPy.
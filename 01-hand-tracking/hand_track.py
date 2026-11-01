import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

CAMERA_INDEX = 1  # try 0 if you get a black window
cap = cv2.VideoCapture(CAMERA_INDEX)

# Palette colors are BGR (OpenCV order)
PALETTE = [
    ("Ocean blue", (190, 119, 0)),
    ("Pink", (180, 105, 255)),
    ("Green", (0, 200, 0)),
]
BOX_X0, BOX_Y0, BOX_W, BOX_H, BOX_GAP = 20, 10, 110, 70, 15
BRUSH = 6
ERASER_R = 40
color = PALETTE[0][1]

SAVE_DIR = Path(__file__).parent / "drawings"
SAVE_DIR.mkdir(exist_ok=True)

ink = None    # the colors that were drawn
mask = None   # where ink exists (lets us draw black too)
prev = None
saved_until = 0


def fingers_up(lm):
    # tips: index 8, middle 12, ring 16, pinky 20; joints are tip-2
    return [lm[t].y < lm[t - 2].y for t in (8, 12, 16, 20)]


def box_at(x, y):
    """Return the palette index under the point (x, y), or None."""
    if not (BOX_Y0 <= y <= BOX_Y0 + BOX_H):
        return None
    for i in range(len(PALETTE)):
        x0 = BOX_X0 + i * (BOX_W + BOX_GAP)
        if x0 <= x <= x0 + BOX_W:
            return i
    return None


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if ink is None:
        ink = np.zeros_like(frame)
        mask = np.zeros((h, w), dtype=np.uint8)

    cursor = None       # fingertip marker
    erase_pos = None    # eraser circle position

    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand = None
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        idx, mid, ring, pinky = fingers_up(lm)
        x, y = int(lm[8].x * w), int(lm[8].y * h)
        cursor = (x, y)

        if idx and not mid and not ring and not pinky:       # draw
            if prev:
                cv2.line(ink, prev, (x, y), color, BRUSH)
                cv2.line(mask, prev, (x, y), 255, BRUSH)
            prev = (x, y)
        elif idx and mid and ring and pinky:                 # open palm = erase
            px, py = int(lm[9].x * w), int(lm[9].y * h)
            cv2.circle(ink, (px, py), ERASER_R, (0, 0, 0), -1)
            cv2.circle(mask, (px, py), ERASER_R, 0, -1)
            erase_pos = (px, py)
            prev = None
        elif idx and mid and not ring and not pinky:         # two fingers = select color
            i = box_at(x, y)
            if i is not None:
                color = PALETTE[i][1]
            prev = None
        elif not any((idx, mid, ring, pinky)):               # fist = clear
            ink[:] = 0
            mask[:] = 0
            prev = None
        else:                                                # anything else = hover
            prev = None
    else:
        prev = None

    # Combine video + drawing
    out = frame.copy()
    out[mask > 0] = ink[mask > 0]

    # Palette
    for i, (name, c) in enumerate(PALETTE):
        x0 = BOX_X0 + i * (BOX_W + BOX_GAP)
        cv2.rectangle(out, (x0, BOX_Y0), (x0 + BOX_W, BOX_Y0 + BOX_H), c, -1)
        border = (255, 255, 255) if c == color else (120, 120, 120)
        cv2.rectangle(out, (x0, BOX_Y0), (x0 + BOX_W, BOX_Y0 + BOX_H),
                      border, 4 if c == color else 1)
        cv2.putText(out, name, (x0 + 8, BOX_Y0 + BOX_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if hand:
        mp_draw.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)
    if cursor:
        cv2.circle(out, cursor, 9, color, -1)
        cv2.circle(out, cursor, 9, (255, 255, 255), 2)
    if erase_pos:
        cv2.circle(out, erase_pos, ERASER_R, (255, 255, 255), 2)
    if time.time() < saved_until:
        cv2.putText(out, "Saved!", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Air Painter (q = quit, s = drawing, p = screenshot)", out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        page = np.full_like(ink, 255)          # white background
        page[mask > 0] = ink[mask > 0]
        path = SAVE_DIR / f"drawing_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), page)
        saved_until = time.time() + 2
        print("Saved", path)
    
    if key == ord("p"):
        path = SAVE_DIR / f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(path), out)
        saved_until = time.time() + 2
        print("Saved", path)    

cap.release()
cv2.destroyAllWindows()
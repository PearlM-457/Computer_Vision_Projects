"""Air Painter: draw in the air with your index finger.

Gestures
    index finger up ............ draw
    index + middle up .......... pick a colour or press a button (point at it)
    open palm (all 4 up) ....... erase under your palm
    fist (hold half a second) .. clear everything
Keys
    q quit | s save drawing on white | p photo | r record video | h shape snap on/off | c clear
Buttons (top bar, press with two fingers or click with the mouse)
    SNAP  turns circles, rectangles, triangles and straight lines into clean shapes
    PHOTO takes a picture of the window   REC starts/stops a video   SAVE saves the drawing on white
Everything is saved in the "drawings" folder next to this script.
"""
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEXES = (1, 0)          # tries camera 1 first, then 0
REQUEST_SIZE = (1280, 720)
VIDEO_FPS = 20
HOLD_TO_CLEAR = 0.6              # seconds a fist must be held
BUTTON_COOLDOWN = 0.9            # seconds before a button can fire again
SMOOTHING = 0.55                 # 1 = raw fingertip, lower = smoother line

# Colours are BGR (OpenCV order). They match the pastel theme of my site.
LIGHT_PINK = (217, 179, 255)     # #ffb3d9
SKY_BLUE = (255, 208, 143)       # #8fd0ff
LILAC = (255, 166, 195)          # #c3a6ff
HOT_PINK = (199, 143, 255)       # #ff8fc7   window frame
DEEP_PINK = (154, 72, 208)       # #d0489a   shadows
INK = (99, 47, 59)               # #3b2f63   text
WHITE = (255, 255, 255)
YELLOW = (122, 226, 255)         # #ffe27a
MINT = (170, 230, 140)
REC_RED = (70, 70, 235)
PALETTE = [("PINK", LIGHT_PINK), ("SKY", SKY_BLUE), ("LILAC", LILAC)]

SAVE_DIR = Path(__file__).parent / "drawings"
FONT = cv2.FONT_HERSHEY_DUPLEX


# ----------------------------------------------------------------------------
# gestures
# ----------------------------------------------------------------------------
def fingers_up(lm):
    """[index, middle, ring, pinky] -> True when the fingertip is above its joint."""
    return [lm[t].y < lm[t - 2].y for t in (8, 12, 16, 20)]


def classify(lm):
    idx, mid, ring, pinky = fingers_up(lm)
    if idx and mid and ring and pinky:
        return "erase"
    if idx and not (mid or ring or pinky):
        return "draw"
    if idx and mid and not (ring or pinky):
        return "select"
    if not (idx or mid or ring or pinky):
        return "clear"
    return "hover"


class Stable:
    """A gesture only counts after it is seen in 3 frames in a row (no flicker)."""

    def __init__(self, n=3):
        self.hist = deque(maxlen=n)
        self.cur = "none"

    def push(self, g):
        self.hist.append(g)
        if len(self.hist) == self.hist.maxlen and len(set(self.hist)) == 1:
            self.cur = g
        return self.cur

    def reset(self):
        self.hist.clear()
        self.cur = "none"


# ----------------------------------------------------------------------------
# shape snapping: turns a rough circle / rectangle / triangle / line into a clean one
# ----------------------------------------------------------------------------
def snap_stroke(pts):
    """Return ("line"|"poly"|"circle"|"ellipse", data) or None to keep the freehand stroke."""
    p = np.asarray(pts, np.float32)
    if len(p) < 10:
        return None
    _, _, bw, bh = cv2.boundingRect(p.astype(np.int32))
    if max(bw, bh) < 45:
        return None
    c = p.reshape(-1, 1, 2)
    length = cv2.arcLength(c, False)
    gap = float(np.linalg.norm(p[0] - p[-1]))

    if gap > 0.22 * length:                       # open stroke: straight line?
        a, b = p[0], p[-1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 60:
            return None
        nrm = np.array([-d[1], d[0]]) / L
        dev = float(np.abs((p - a) @ nrm).max())
        return ("line", (a, b)) if dev < 0.07 * L else None

    # closed stroke: only convex-looking shapes are snapped
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    harea = cv2.contourArea(hull)
    if area < 400 or harea < 1 or area / harea < 0.85:
        return None                               # flower, star, heart...: leave as drawn
    hperi = cv2.arcLength(hull, True)
    candidates = []
    if len(p) >= 5:                               # round: circle or ellipse
        box = cv2.fitEllipse(c)
        (cx, cy), (MA, ma), _ = box
        if max(MA, ma) / max(1e-3, min(MA, ma)) < 1.25:
            candidates.append(("circle", ((cx, cy), (MA + ma) / 4)))
        else:
            candidates.append(("ellipse", box))
    approx = cv2.approxPolyDP(hull, 0.04 * hperi, True).reshape(-1, 2)
    if len(approx) == 3:                          # triangle
        candidates.append(("poly", approx.astype(np.float32)))
    elif len(approx) == 4:                        # rectangle
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(hull)
        if rw * rh > 0 and harea / (rw * rh) > 0.82:
            q = round(ang / 90)
            if abs(ang - q * 90) < 8:             # nearly straight: make it level
                if q % 2:
                    rw, rh = rh, rw
                ang = 0
            candidates.append(("poly", cv2.boxPoints(((cx, cy), (rw, rh), ang))))
        else:
            candidates.append(("poly", approx.astype(np.float32)))
    best = max(((_iou(c, sh), i, sh) for i, sh in enumerate(candidates)), default=None, key=lambda t: t[0])
    if best is None or best[0] < 0.88:
        return None                               # not close enough to a clean shape
    return best[2]


def _iou(contour, shape):
    """How much the drawn outline and the clean shape overlap (0 to 1)."""
    pts = contour.reshape(-1, 2)
    off = np.floor(pts.min(0)) - 30
    size = (int(np.ceil(pts.max(0)[0] - off[0])) + 30, int(np.ceil(pts.max(0)[1] - off[1])) + 30)
    a = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(a, [np.round(pts - off).astype(np.int32).reshape(-1, 1, 2)], 1)
    b = np.zeros_like(a)
    kind, data = shape
    if kind == "poly":
        cv2.fillPoly(b, [np.round(np.asarray(data) - off).astype(np.int32).reshape(-1, 1, 2)], 1)
    elif kind == "circle":
        (cx, cy), r = data
        cv2.circle(b, (int(cx - off[0]), int(cy - off[1])), int(r), 1, -1)
    elif kind == "ellipse":
        (cx, cy), axes, ang = data
        cv2.ellipse(b, ((cx - off[0], cy - off[1]), axes, ang), 1, -1)
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum()) / union if union else 0.0


def draw_shape(img, shape, color, thick):
    kind, data = shape
    aa = cv2.LINE_AA
    if kind == "line":
        a, b = data
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, thick, aa)
    elif kind == "poly":
        cv2.polylines(img, [np.round(data).astype(np.int32).reshape(-1, 1, 2)], True, color, thick, aa)
    elif kind == "circle":
        (cx, cy), r = data
        cv2.circle(img, (int(cx), int(cy)), int(r), color, thick, aa)
    elif kind == "ellipse":
        cv2.ellipse(img, data, color, thick, aa)


# ----------------------------------------------------------------------------
# the canvas
# ----------------------------------------------------------------------------
def over(ink, mask, tink, tmask):
    """Put a stroke layer (tink, tmask) on top of the canvas (ink, mask)."""
    tm = tmask.astype(np.float32) / 255.0
    new_ink = tink.astype(np.float32) + ink.astype(np.float32) * (1 - tm[..., None])
    new_mask = tmask.astype(np.float32) + mask.astype(np.float32) * (1 - tm)
    return np.clip(new_ink, 0, 255).astype(np.uint8), np.clip(new_mask, 0, 255).astype(np.uint8)


class Painter:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.brush = max(6, int(w * 0.0095))
        self.color = LIGHT_PINK
        self.ink = np.zeros((h, w, 3), np.uint8)   # colour, already multiplied by coverage
        self.mask = np.zeros((h, w), np.uint8)     # how much ink is there (soft edges)
        self.tink = np.zeros_like(self.ink)        # the stroke being drawn right now
        self.tmask = np.zeros_like(self.mask)
        self.pts = []

    def add_point(self, p):
        p = (int(p[0]), int(p[1]))
        if self.pts:
            a = self.pts[-1]
            cv2.line(self.tink, a, p, self.color, self.brush, cv2.LINE_AA)
            cv2.line(self.tmask, a, p, 255, self.brush, cv2.LINE_AA)
        else:
            cv2.circle(self.tink, p, self.brush // 2, self.color, -1, cv2.LINE_AA)
            cv2.circle(self.tmask, p, self.brush // 2, 255, -1, cv2.LINE_AA)
        self.pts.append(p)

    def finish(self, snap=True):
        """End the stroke. Returns the name of the shape it was snapped to, or None."""
        if not self.pts:
            return None
        label = None
        shape = snap_stroke(self.pts) if snap else None
        if shape:
            draw_shape(self.ink, shape, self.color, self.brush)
            sm = np.zeros_like(self.mask)
            draw_shape(sm, shape, 255, self.brush)
            self.mask = np.maximum(self.mask, sm)
            label = {"poly": "shape", "line": "line"}.get(shape[0], shape[0])
            if shape[0] == "poly":
                label = {3: "triangle", 4: "rectangle"}.get(len(shape[1]), "shape")
        else:
            self.ink, self.mask = over(self.ink, self.mask, self.tink, self.tmask)
        self.tink[:] = 0
        self.tmask[:] = 0
        self.pts = []
        return label

    def erase(self, center, r):
        c = (int(center[0]), int(center[1]))
        for layer, val in ((self.ink, (0, 0, 0)), (self.mask, 0)):
            cv2.circle(layer, c, r, val, -1)

    def clear(self):
        for a in (self.ink, self.mask, self.tink, self.tmask):
            a[:] = 0
        self.pts = []

    def layers(self):
        if self.pts:
            return over(self.ink, self.mask, self.tink, self.tmask)
        return self.ink, self.mask

    def compose(self, frame):
        ink, mask = self.layers()
        m = (mask.astype(np.float32) / 255.0)[..., None]
        out = frame.astype(np.float32) * (1 - m) + ink.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def on_white(self):
        ink, mask = self.layers()
        m = (mask.astype(np.float32) / 255.0)[..., None]
        return np.clip(255 * (1 - m) + ink.astype(np.float32), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# arcade-style window: pink frame, pixel buttons, colour row on top
# ----------------------------------------------------------------------------
def make_layout(w, h):
    B = max(8, w // 80)                           # frame thickness
    bar = max(76, int(h * 0.15))                  # top bar height
    strip = max(28, int(h * 0.06)) + B            # bottom strip height (includes the frame)
    pad = B
    y0 = B + pad
    ih = bar - y0 - pad
    gap = max(8, w // 90)
    x = B * 3 + 10                                # clear of the corner block
    items = []
    for i, (name, col) in enumerate(PALETTE):
        items.append({"name": name, "kind": "color", "color": col, "rect": (x, y0, x + ih, y0 + ih)})
        x += ih + gap
    x += gap * 2
    bw = max(86, int(w * 0.115))
    for name, kind in (("SNAP", "shape"), ("PHOTO", "photo"), ("REC", "rec"), ("SAVE", "save")):
        items.append({"name": name, "kind": kind, "rect": (x, y0, x + bw, y0 + ih)})
        x += bw + gap
    return {"B": B, "bar": bar, "strip": strip, "ih": ih, "items": items}


def hit(layout, x, y):
    for it in layout["items"]:
        x0, y0, x1, y1 = it["rect"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return it
    return None


def put_center(img, text, cx, cy, scale, color, thick=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)), FONT, scale, color, thick, cv2.LINE_AA)


def pixel_button(img, rect, fill, label="", hover=False, active=False, scale=0.5):
    x0, y0, x1, y1 = rect
    cv2.rectangle(img, (x0 + 4, y0 + 4), (x1 + 4, y1 + 4), DEEP_PINK, -1)      # arcade drop shadow
    cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
    cv2.line(img, (x0 + 3, y0 + 3), (x1 - 3, y0 + 3), WHITE, 1)               # top shine
    cv2.rectangle(img, (x0, y0), (x1, y1), INK, 2)
    if active:
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), WHITE, 3)
    if hover:
        cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), YELLOW, 4)
    if label:
        put_center(img, label, (x0 + x1) / 2, (y0 + y1) / 2, scale, INK, 1)


def blend_rect(img, x0, y0, x1, y1, color, alpha):
    roi = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)


def draw_ui(out, layout, color, snap_on, recording, hover_name, rec_secs, blink):
    h, w = out.shape[:2]
    B, bar, strip, ih = layout["B"], layout["bar"], layout["strip"], layout["ih"]
    blend_rect(out, 0, 0, w, bar, LIGHT_PINK, 0.8)                           # top bar
    cv2.line(out, (0, bar), (w, bar), DEEP_PINK, 3)
    blend_rect(out, 0, h - strip, w, h, LIGHT_PINK, 0.8)                     # bottom strip
    cv2.line(out, (0, h - strip), (w, h - strip), DEEP_PINK, 3)

    fs = max(0.4, ih / 100.0)
    for it in layout["items"]:
        hov = it["name"] == hover_name
        if it["kind"] == "color":
            pixel_button(out, it["rect"], it["color"], "", hover=hov, active=(it["color"] == color))
        else:
            fill, label = WHITE, it["name"]
            if it["kind"] == "shape":
                fill, label = (MINT if snap_on else WHITE), ("SNAP ON" if snap_on else "SNAP OFF")
            if it["kind"] == "rec" and recording:
                fill, label = REC_RED, "STOP"
            pixel_button(out, it["rect"], fill, label, hover=hov, scale=fs)

    ty = h - (strip + B) // 2 + 6                  # text sits in the visible part of the strip
    cv2.putText(out, "AIR PAINTER", (B * 3 + 12, ty), FONT, 0.6, INK, 1, cv2.LINE_AA)
    hint = ("1 finger draw  |  2 pick  |  palm erase  |  fist clear" if w < 900
            else "INDEX draw   |   2 FINGERS pick   |   PALM erase   |   FIST clear")
    (tw, _), _ = cv2.getTextSize(hint, FONT, 0.5, 1)
    cv2.putText(out, hint, (w - tw - B * 3 - 12, ty - 1), FONT, 0.5, INK, 1, cv2.LINE_AA)

    if recording and blink:                                                   # REC dot + timer
        cv2.circle(out, (w - B - 120, bar + 26), 9, REC_RED, -1, cv2.LINE_AA)
    if recording:
        t = f"REC {int(rec_secs) // 60:02d}:{int(rec_secs) % 60:02d}"
        cv2.putText(out, t, (w - B - 100, bar + 33), FONT, 0.6, WHITE, 3, cv2.LINE_AA)
        cv2.putText(out, t, (w - B - 100, bar + 33), FONT, 0.6, REC_RED, 1, cv2.LINE_AA)

    # square pink frame with chunky corner blocks
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), HOT_PINK, B * 2)
    cv2.rectangle(out, (B, B), (w - 1 - B, h - 1 - B), WHITE, 2)
    c = B * 3
    for (x0, y0), col in (((0, 0), SKY_BLUE), ((w - c, 0), LILAC), ((0, h - c), LILAC), ((w - c, h - c), SKY_BLUE)):
        cv2.rectangle(out, (x0, y0), (x0 + c - 1, y0 + c - 1), col, -1)
        cv2.rectangle(out, (x0, y0), (x0 + c - 1, y0 + c - 1), INK, 2)


def draw_toast(out, text, layout):
    h, w = out.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cx, cy = w // 2, h - layout["strip"] - 30
    pixel_button(out, (cx - tw // 2 - 16, cy - th - 8, cx + tw // 2 + 16, cy + 10), WHITE, text, scale=0.6)


# ----------------------------------------------------------------------------
# main loop
# ----------------------------------------------------------------------------
def open_camera():
    for i in CAMERA_INDEXES:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_SIZE[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_SIZE[1])
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def main():
    import mediapipe as mp                     # imported here so the helpers above can be tested without it

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

    cap = open_camera()
    if cap is None:
        print("Could not open a camera. Change CAMERA_INDEXES at the top of the file.")
        return
    SAVE_DIR.mkdir(exist_ok=True)
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    painter = Painter(w, h)
    layout = make_layout(w, h)
    stamp = lambda: time.strftime("%Y%m%d_%H%M%S")

    win = "Air Painter"
    click = {"pos": None}
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: click.update(pos=(x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    stab = Stable(3)
    smooth = None
    snap_on = True
    writer, rec_start, next_write = None, 0.0, 0.0
    cooldown = {}
    toast, toast_until, flash_until = "", 0.0, 0.0
    fist_since, fist_done = None, False
    want_photo = False

    def say(text, secs=1.8):
        nonlocal toast, toast_until
        toast, toast_until = text, time.time() + secs

    def press(item, now, key=False):
        nonlocal snap_on, want_photo, writer, rec_start, next_write
        if item["kind"] == "color":
            painter.color = item["color"]
            return
        if not key:                                  # keyboard presses skip the cooldown
            if now < cooldown.get(item["name"], 0):
                return
            cooldown[item["name"]] = now + BUTTON_COOLDOWN
        if item["kind"] == "shape":
            snap_on = not snap_on
            say("Shape snap ON" if snap_on else "Shape snap OFF")
        elif item["kind"] == "photo":
            want_photo = True
        elif item["kind"] == "save":
            path = SAVE_DIR / f"drawing_{stamp()}.png"
            cv2.imwrite(str(path), painter.on_white())
            say("Saved drawing!")
            print("Saved", path)
        elif item["kind"] == "rec":
            if writer is None:
                path = SAVE_DIR / f"video_{stamp()}.mp4"
                wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
                if wr.isOpened():
                    writer, rec_start, next_write = wr, now, now
                    print("Recording", path)
                else:
                    say("Could not start recording")
            else:
                writer.release()
                writer = None
                say("Video saved!")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        now = time.time()

        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand, gesture, cursor, palm = None, "none", None, None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            raw_g = classify(lm)
            gesture = stab.push(raw_g)
            if gesture != raw_g:                       # still changing: do nothing until it settles
                gesture = "hover"
            raw = np.array([lm[8].x * w, lm[8].y * h])
            smooth = raw if smooth is None else SMOOTHING * raw + (1 - SMOOTHING) * smooth
            cursor = (int(smooth[0]), int(smooth[1]))
            palm = (int(lm[9].x * w), int(lm[9].y * h))
        else:
            stab.reset()
            smooth = None

        B = layout["B"]
        can_draw = (cursor is not None and layout["bar"] + 6 < cursor[1] < h - layout["strip"] - 6
                    and B + 4 < cursor[0] < w - B - 4)
        hover = hit(layout, *cursor) if cursor else None

        # drawing
        if gesture == "draw" and can_draw:
            if not painter.pts:                        # new stroke starts exactly at the fingertip
                smooth = raw
                cursor = (int(raw[0]), int(raw[1]))
            painter.add_point(cursor)
        elif painter.pts:
            label = painter.finish(snap_on)
            if label:
                say(f"Snapped to {label}")

        # erase with the open palm
        erase_r = int(w * 0.055)
        if gesture == "erase" and palm:
            painter.erase(palm, erase_r)

        # fist held = clear all
        progress = 0.0
        if gesture == "clear":
            if fist_since is None:
                fist_since = now
            progress = min(1.0, (now - fist_since) / HOLD_TO_CLEAR)
            if progress >= 1.0 and not fist_done:
                painter.clear()
                fist_done = True
                say("Cleared!")
        else:
            fist_since, fist_done = None, False

        # two fingers on a colour / button
        if gesture == "select" and hover:
            press(hover, now)
        if click["pos"]:
            item = hit(layout, *click["pos"])
            click["pos"] = None
            if item:
                press(item, now)

        # ---- build the picture ----
        out = painter.compose(frame)
        if hand:
            mp_draw.draw_landmarks(
                out, hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=WHITE, thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=HOT_PINK, thickness=2))
        if cursor:
            if gesture == "draw":
                cv2.circle(out, cursor, 10, painter.color, -1, cv2.LINE_AA)
                cv2.circle(out, cursor, 10, WHITE, 2, cv2.LINE_AA)
            elif gesture == "select":
                cv2.circle(out, cursor, 18, YELLOW, 3, cv2.LINE_AA)
            elif gesture == "erase" and palm:
                cv2.circle(out, palm, erase_r, WHITE, 3, cv2.LINE_AA)
                cv2.circle(out, palm, erase_r, HOT_PINK, 1, cv2.LINE_AA)
            elif gesture == "clear" and progress > 0:
                cv2.ellipse(out, palm or cursor, (30, 30), -90, 0, 360 * progress, DEEP_PINK, 6, cv2.LINE_AA)
            else:
                cv2.circle(out, cursor, 8, WHITE, 2, cv2.LINE_AA)
        rec_secs = (now - rec_start) if writer else 0
        draw_ui(out, layout, painter.color, snap_on, writer is not None,
                hover.get("name") if hover and cursor else None, rec_secs, int(now * 2) % 2 == 0)

        # photo / video get the picture without the pop-ups
        if want_photo:
            path = SAVE_DIR / f"photo_{stamp()}.png"
            cv2.imwrite(str(path), out)
            flash_until = now + 0.25
            say("Photo saved!")
            print("Saved", path)
            want_photo = False
        if writer:
            while next_write <= now:
                writer.write(out)
                next_write += 1.0 / VIDEO_FPS
            if now - next_write > 1:
                next_write = now

        if now < toast_until:
            draw_toast(out, toast, layout)
        if now < flash_until:
            a = 0.8 * (flash_until - now) / 0.25
            out = cv2.addWeighted(out, 1 - a, np.full_like(out, 255), a, 0)

        cv2.imshow(win, out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            painter.clear()
        if key == ord("h"):
            press({"kind": "shape", "name": "SNAP"}, now, key=True)
        if key == ord("p"):
            press({"kind": "photo", "name": "PHOTO"}, now, key=True)
        if key == ord("s"):
            press({"kind": "save", "name": "SAVE"}, now, key=True)
        if key == ord("r"):
            press({"kind": "rec", "name": "REC"}, now, key=True)

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

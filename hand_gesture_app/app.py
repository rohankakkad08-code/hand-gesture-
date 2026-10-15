# -*- coding: utf-8 -*-
"""
app.py
Flask server — streams annotated webcam frames and exposes a JSON
endpoint for the latest detected gesture.

On cloud/Render (no physical camera) the video feed shows a static
"No Camera" placeholder and the gesture endpoint returns a demo state.
"""

import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template
from gesture_engine import classify

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock            = threading.Lock()
_latest_gesture  = {"name": "No Hand Detected", "emoji": "✋", "color": "#6b7280", "confidence": 0}
_frame_bytes     = None          # JPEG bytes of the latest annotated frame

# ── Colours / drawing helpers ─────────────────────────────────────────────────
LANDMARK_COLOR   = (0, 255, 200)
CONNECTION_COLOR = (100, 220, 255)
BBOX_COLOR       = (0, 200, 255)
FONT             = cv2.FONT_HERSHEY_SIMPLEX


def _draw_landmarks(image, hand_landmarks, h, w):
    """Draw custom landmarks and connections."""
    mp_draw         = mp.solutions.drawing_utils
    mp_hands_module = mp.solutions.hands

    mp_draw.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands_module.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=LANDMARK_COLOR, thickness=2, circle_radius=4),
        mp_draw.DrawingSpec(color=CONNECTION_COLOR, thickness=2),
    )


def _bounding_box(landmarks, h, w):
    xs  = [int(lm.x * w) for lm in landmarks.landmark]
    ys  = [int(lm.y * h) for lm in landmarks.landmark]
    pad = 20
    return (
        max(min(xs) - pad, 0),
        max(min(ys) - pad, 0),
        min(max(xs) + pad, w),
        min(max(ys) + pad, h),
    )


def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b   = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (b, g, r)


def _make_no_camera_frame():
    """Generate a static 1280×720 placeholder frame (no webcam available)."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (10, 10, 20)
    cv2.putText(frame, "No Camera Available",        (360, 340), FONT, 1.8, (0, 255, 200), 3)
    cv2.putText(frame, "Running in demo/cloud mode", (400, 400), FONT, 1.0, (100, 100, 120), 2)
    cv2.putText(frame, "Hand Gesture Recognition",   (12, 40),   FONT, 1.0, (0, 255, 200), 2)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


# ── Camera thread ─────────────────────────────────────────────────────────────
def _camera_loop():
    global _frame_bytes, _latest_gesture

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # No camera — serve a static placeholder frame forever
        placeholder = _make_no_camera_frame()
        with _lock:
            _frame_bytes = placeholder
        while True:
            time.sleep(60)   # keep thread alive, frame never changes

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Glassy dark overlay for aesthetic ────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)

        best_gesture = {"name": "No Hand Detected", "emoji": "✋", "color": "#6b7280", "confidence": 0}

        if result.multi_hand_landmarks:
            for idx, hand_lm in enumerate(result.multi_hand_landmarks):
                _draw_landmarks(frame, hand_lm, h, w)

                # Bounding box
                x1, y1, x2, y2    = _bounding_box(hand_lm, h, w)
                gesture_name, emoji, color = classify(hand_lm.landmark)
                bgr = _hex_to_bgr(color)

                # Glowing box
                for thickness in [6, 3]:
                    alpha = 0.4 if thickness == 6 else 1.0
                    tmp   = frame.copy()
                    cv2.rectangle(tmp, (x1, y1), (x2, y2), bgr, thickness)
                    cv2.addWeighted(tmp, alpha, frame, 1 - alpha, 0, frame)

                # Gesture label on box
                label       = f"  {gesture_name}  "
                (tw, th), _ = cv2.getTextSize(label, FONT, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw, y1), bgr, -1)
                cv2.putText(frame, label, (x1, y1 - 6), FONT, 0.7, (255, 255, 255), 2)

                # Hand label (Left / Right)
                if result.multi_handedness:
                    handedness = result.multi_handedness[idx].classification[0].label
                    cv2.putText(frame, handedness, (x1, y2 + 22), FONT, 0.55, bgr, 2)

                if idx == 0:
                    best_gesture = {
                        "name":       gesture_name,
                        "emoji":      emoji,
                        "color":      color,
                        "confidence": round(
                            result.multi_handedness[idx].classification[0].score * 100
                        ) if result.multi_handedness else 90,
                    }

        # Header bar text
        cv2.putText(frame, "Hand Gesture Recognition", (12, 40), FONT, 1.0, (0, 255, 200), 2)

        with _lock:
            _latest_gesture = best_gesture
            _, buf          = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _frame_bytes    = buf.tobytes()

    cap.release()
    mp_hands.close()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def _gen():
        while True:
            with _lock:
                data = _frame_bytes
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(0.03)

    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/gesture")
def gesture():
    with _lock:
        return jsonify(_latest_gesture)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[OK] Server running -> http://127.0.0.1:{port}\n")
    try:
        from waitress import serve
        print("[INFO] Using waitress server (Windows)\n")
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

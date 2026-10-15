# -*- coding: utf-8 -*-
"""
gesture_engine.py
Core hand-gesture recognition logic using MediaPipe Hands.
Detects: thumbs_up, thumbs_down, wave, fist, open_hand,
         peace, rock, ok, pointing, front_facing, back_facing
"""

import math
import mediapipe as mp

mp_hands = mp.solutions.hands

# ── landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_extended(lm, tip, pip):
    """True when fingertip is above (lower y value) its PIP joint."""
    return lm[tip].y < lm[pip].y


def _thumb_extended(lm):
    """Thumb is extended when tip is far from the base of index finger."""
    return _dist(lm[THUMB_TIP], lm[INDEX_MCP]) > _dist(lm[THUMB_IP], lm[INDEX_MCP])


def classify(landmarks):
    """
    Given a list of 21 NormalizedLandmark objects, return a (name, emoji, color)
    tuple for the best-matching gesture.
    """
    lm = landmarks

    index_up  = _finger_extended(lm, INDEX_TIP,  INDEX_PIP)
    middle_up = _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_up   = _finger_extended(lm, RING_TIP,   RING_PIP)
    pinky_up  = _finger_extended(lm, PINKY_TIP,  PINKY_PIP)
    thumb_up  = _thumb_extended(lm)

    fingers_up = [index_up, middle_up, ring_up, pinky_up]
    n_up = sum(fingers_up)

    # ── Thumbs-up ──────────────────────────────────────────────────────────────
    # Thumb tip clearly above wrist, all other fingers curled
    thumb_above_wrist = lm[THUMB_TIP].y < lm[WRIST].y - 0.05
    if thumb_above_wrist and n_up == 0:
        return ("Thumbs Up", "👍", "#22c55e")

    # ── Thumbs-down ────────────────────────────────────────────────────────────
    thumb_below_wrist = lm[THUMB_TIP].y > lm[WRIST].y + 0.05
    if thumb_below_wrist and n_up == 0:
        return ("Thumbs Down", "👎", "#ef4444")

    # ── Fist ───────────────────────────────────────────────────────────────────
    if n_up == 0 and not thumb_up:
        return ("Fist", "✊", "#f97316")

    # ── Open Hand ──────────────────────────────────────────────────────────────
    if n_up == 4 and thumb_up:
        return ("Open Hand", "🖐", "#3b82f6")

    # ── Peace / Victory ────────────────────────────────────────────────────────
    if index_up and middle_up and not ring_up and not pinky_up:
        return ("Peace ✌", "✌️", "#a855f7")

    # ── Pointing ───────────────────────────────────────────────────────────────
    if index_up and not middle_up and not ring_up and not pinky_up:
        return ("Pointing", "☝️", "#06b6d4")

    # ── Rock / Devil Horns ─────────────────────────────────────────────────────
    if index_up and pinky_up and not middle_up and not ring_up:
        return ("Rock On 🤘", "🤘", "#ec4899")

    # ── OK gesture ─────────────────────────────────────────────────────────────
    thumb_index_close = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.06
    if thumb_index_close and middle_up and ring_up and pinky_up:
        return ("OK 👌", "👌", "#eab308")

    # ── Wave (spread fingers, wrist level) ─────────────────────────────────────
    if n_up == 4:
        spread = (
            _dist(lm[INDEX_TIP], lm[PINKY_TIP]) > 0.30
        )
        if spread:
            return ("Wave 👋", "👋", "#f59e0b")
        return ("Open Hand", "🖐", "#3b82f6")

    # ── Three fingers ──────────────────────────────────────────────────────────
    if index_up and middle_up and ring_up and not pinky_up:
        return ("Three", "3️⃣", "#14b8a6")

    # ── Front / Back ───────────────────────────────────────────────────────────
    # Approximate by checking if knuckles face camera (landmarks z depth proxy)
    # MediaPipe z is relative depth; negative = closer to camera
    avg_tip_z    = (lm[INDEX_TIP].z + lm[MIDDLE_TIP].z + lm[RING_TIP].z) / 3
    avg_mcp_z    = (lm[INDEX_MCP].z + lm[MIDDLE_MCP].z + lm[RING_MCP].z) / 3
    palm_forward = avg_tip_z < avg_mcp_z   # tips closer than knuckles → palm faces cam

    if n_up >= 3:
        if palm_forward:
            return ("Front Hand", "🤚", "#64748b")
        else:
            return ("Back Hand", "🫲", "#94a3b8")

    return ("Unknown", "❓", "#6b7280")

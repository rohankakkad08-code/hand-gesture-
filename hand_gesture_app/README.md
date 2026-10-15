# 🤚 Hand Gesture Recognition

Real-time hand gesture recognition app using **MediaPipe**, **OpenCV**, and **Flask**, with a polished dark-themed HTML/CSS frontend.

---

## ✨ Recognized Gestures

| Gesture | Emoji | Description |
|---|---|---|
| Thumbs Up | 👍 | Thumb extended upward, all fingers curled |
| Thumbs Down | 👎 | Thumb pointing downward, all fingers curled |
| Fist | ✊ | All fingers curled |
| Open Hand | 🖐 | All five fingers extended |
| Peace | ✌️ | Index + middle finger extended |
| Pointing | ☝️ | Only index finger extended |
| Rock On | 🤘 | Index + pinky extended |
| OK | 👌 | Thumb + index form a circle, others extended |
| Wave | 👋 | All 4 fingers extended and spread wide |
| Front Hand | 🤚 | Palm facing camera |
| Back Hand | 🫲 | Back of hand facing camera |
| Three Fingers | 3️⃣ | Index + middle + ring extended |

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
cd hand_gesture_app
pip install -r requirements.txt
```

### 2 — Run the server

```bash
python app.py
```

### 3 — Open in browser

```
http://127.0.0.1:5000
```

---

## 📁 File Structure

```
hand_gesture_app/
├── app.py              ← Flask server + camera thread
├── gesture_engine.py   ← Pure gesture classification logic
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Attractive dark-themed frontend
└── README.md
```

---

## 🛠 How It Works

```
Webcam → OpenCV frame → MediaPipe Hands (21 landmarks)
       → gesture_engine.classify() → gesture name / emoji / colour
       → Annotated JPEG streamed via /video_feed  (MJPEG)
       → /gesture  JSON endpoint polled at 15 Hz by browser
```

- **gesture_engine.py** uses only landmark geometry (finger up/down, distances, z-depth) — no ML model beyond MediaPipe's built-in hand detector.
- The Flask server runs the camera loop in a background thread; the main thread serves HTTP requests.
- The frontend polls `/gesture` at ~15 Hz and updates the card, history, and stats without reloading.

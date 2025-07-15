# 🤟 Sign Language Recognizer

A real-time sign language recognizer built with MediaPipe and OpenCV. It detects hand landmarks via webcam input and classifies gestures (A–Z) using a custom-trained keypoint classifier.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/zGfgGZuizO0/0.jpg)](https://youtu.be/zGfgGZuizO0)

---

## 🚀 Features
- Real-time hand tracking and gesture recognition
- Supports 2-hand detection
- Custom gesture logging for training
- Simple overlay with gesture label and hand orientation

---

## 🛠 Tech Stack
- Python 3
- OpenCV
- MediaPipe
- NumPy

---

## ▶️ How to Run
```bash
pip install opencv-python mediapipe numpy
python main.py
```

## 🔁 Logging New Gestures
Press 1 to enter logging mode
Press a letter key (A–Z) to log that gesture
Data is saved in model/keypoint.csv for training

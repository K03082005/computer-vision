# 🖐️ Volume Control Using Hand Gestures (Computer Vision)



Control your system volume in real-time using just your hand — no mouse, no keyboard. Built with Python, OpenCV, and MediaPipe.

---

## 📸 Demo

| Gesture | Action |
|--------|--------|
| Thumb & Index far apart | 🔊 Volume Up |
| Thumb & Index close together | 🔇 Volume Down / Mute |

---

## 🧠 How It Works

1. Webcam captures live video frames
2. **MediaPipe** detects 21 hand landmarks in real-time
3. Distance between **thumb tip** and **index finger tip** is calculated
4. Distance is **mapped** to system volume range (0% → 100%)
5. **pycaw** sets the system audio volume accordingly
6. FPS counter and volume bar displayed on screen via **OpenCV**

---

## 🛠️ Tech Stack

| Library | Purpose |
|--------|---------|
| `opencv-python` | Webcam capture & drawing |
| `mediapipe` | Hand landmark detection |
| `pycaw` | Windows audio control |
| `numpy` | Distance interpolation |
| `comtypes` | COM interface for audio |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 (recommended)
- Windows OS (pycaw is Windows-only)
- Webcam

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/K03082005/computer-vision.git
cd computer-vision

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install opencv-python mediapipe numpy pycaw comtypes
```

---



## 📁 Project Structure

```
computer-vision/
│
├── volume_hand_control.py   # Main script (volume control)
├── hand_control_module.py   # Hand tracking module (reusable)
├── README.md                # Project documentation
└── requirements.txt         # Dependencies
```

---

## 📦 requirements.txt

```
opencv-python
mediapipe
numpy
pycaw
comtypes
```



---

## 📊 Results

- ✅ Real-time hand detection at **25–30 FPS**
- ✅ Volume control response time: **< 100ms**
- ✅ Works in normal indoor lighting
- ⚠️ Accuracy drops in low light or fast movement

---


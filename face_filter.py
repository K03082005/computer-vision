import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import time

# ── Download face landmarker model ──
if not os.path.exists("face_landmarker.task"):
    print("Downloading face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker.task"
    )
    print("Done!")

# ── Create filter images using OpenCV (no external images needed) ──
def createHat(w=200, h=150):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # Brim
    cv2.ellipse(img, (w//2, h-20), (w//2, 25), 0, 0, 360, (20, 20, 20, 255), -1)
    # Top
    pts = np.array([[w//4, h-20], [3*w//4, h-20], [2*w//3, 10], [w//3, 10]], np.int32)
    cv2.fillPoly(img, [pts], (30, 30, 30, 255))
    # Band
    cv2.rectangle(img, (w//4, h-35), (3*w//4, h-25), (180, 50, 50, 255), -1)
    return img

def createGlasses(w=300, h=100):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # Left lens
    cv2.ellipse(img, (w//4, h//2), (w//5, h//3), 0, 0, 360, (50, 50, 200, 200), -1)
    cv2.ellipse(img, (w//4, h//2), (w//5, h//3), 0, 0, 360, (20, 20, 20, 255), 3)
    # Right lens
    cv2.ellipse(img, (3*w//4, h//2), (w//5, h//3), 0, 0, 360, (50, 50, 200, 200), -1)
    cv2.ellipse(img, (3*w//4, h//2), (w//5, h//3), 0, 0, 360, (20, 20, 20, 255), 3)
    # Bridge
    cv2.line(img, (w//4 + w//5, h//2), (3*w//4 - w//5, h//2), (20, 20, 20, 255), 3)
    # Arms
    cv2.line(img, (w//4 - w//5, h//2), (0, h//2 - 10), (20, 20, 20, 255), 3)
    cv2.line(img, (3*w//4 + w//5, h//2), (w, h//2 - 10), (20, 20, 20, 255), 3)
    return img

def createSunglasses(w=300, h=100):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # Left lens
    cv2.ellipse(img, (w//4, h//2), (w//5, h//3), 0, 0, 360, (0, 50, 0, 220), -1)
    cv2.ellipse(img, (w//4, h//2), (w//5, h//3), 0, 0, 360, (0, 0, 0, 255), 3)
    # Right lens
    cv2.ellipse(img, (3*w//4, h//2), (w//5, h//3), 0, 0, 360, (0, 50, 0, 220), -1)
    cv2.ellipse(img, (3*w//4, h//2), (w//5, h//3), 0, 0, 360, (0, 0, 0, 255), 3)
    # Bridge
    cv2.line(img, (w//4 + w//5, h//2), (3*w//4 - w//5, h//2), (0, 0, 0, 255), 3)
    # Arms
    cv2.line(img, (w//4 - w//5, h//2), (0, h//2 - 10), (0, 0, 0, 255), 3)
    cv2.line(img, (3*w//4 + w//5, h//2), (w, h//2 - 10), (0, 0, 0, 255), 3)
    return img

def createCrown(w=200, h=120):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # Base
    pts = np.array([[0, h], [0, h//2], [w//4, h//4],
                    [w//2, h//2], [3*w//4, h//4],
                    [w, h//2], [w, h]], np.int32)
    cv2.fillPoly(img, [pts], (0, 180, 255, 255))
    # Gems
    cv2.circle(img, (w//4, h//4 + 10), 8, (0, 0, 255, 255), -1)
    cv2.circle(img, (w//2, h//2 - 5), 8, (255, 0, 255, 255), -1)
    cv2.circle(img, (3*w//4, h//4 + 10), 8, (0, 0, 255, 255), -1)
    # Outline
    cv2.polylines(img, [pts], False, (0, 120, 200, 255), 2)
    return img

def createMustache(w=160, h=60):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    # Left side
    cv2.ellipse(img, (w//4, h//2), (w//4, h//3), 20, 0, 180, (30, 20, 10, 255), -1)
    # Right side
    cv2.ellipse(img, (3*w//4, h//2), (w//4, h//3), -20, 0, 180, (30, 20, 10, 255), -1)
    return img

def createHeart(w=100, h=100):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    cv2.circle(img, (w//4, h//3), w//5, (0, 0, 255, 255), -1)
    cv2.circle(img, (3*w//4, h//3), w//5, (0, 0, 255, 255), -1)
    pts = np.array([[0, h//3], [w//2, h], [w, h//3]], np.int32)
    cv2.fillPoly(img, [pts], (0, 0, 255, 255))
    return img

# ── All filters ──
filters = {
    "Hat":         createHat(200, 150),
    "Glasses":     createGlasses(300, 100),
    "Sunglasses":  createSunglasses(300, 100),
    "Crown":       createCrown(200, 120),
    "Mustache":    createMustache(160, 60),
}
filterNames = list(filters.keys())
currentFilter = 0
showAll = False

# ── MediaPipe setup ──
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=3,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face landmark indices
FOREHEAD     = 10
LEFT_EYE     = 130
RIGHT_EYE    = 359
NOSE_TIP     = 4
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
CHIN         = 152
LEFT_TEMPLE  = 234
RIGHT_TEMPLE = 454
NOSE_BOTTOM  = 2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
screenshotCount = 0


def overlayPNG(bg, overlay, x, y, w, h):
    """Overlay a PNG with alpha channel onto background"""
    try:
        overlay_resized = cv2.resize(overlay, (w, h))

        # Clamp to image bounds
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])

        if x2 <= x1 or y2 <= y1:
            return bg

        # Crop overlay to fit
        ox1 = x1 - x
        oy1 = y1 - y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        alpha = overlay_resized[oy1:oy2, ox1:ox2, 3:4] / 255.0
        color = overlay_resized[oy1:oy2, ox1:ox2, :3]

        bg[y1:y2, x1:x2] = (bg[y1:y2, x1:x2] * (1 - alpha) + color * alpha).astype(np.uint8)
    except Exception:
        pass
    return bg


def applyFilter(img, landmarks, filterName, w, h):
    lm = landmarks

    if filterName == "Hat":
        # Position above forehead
        top    = lm[FOREHEAD]
        left   = lm[LEFT_TEMPLE]
        right  = lm[RIGHT_TEMPLE]
        faceW  = int(abs(right.x - left.x) * w * 1.4)
        faceH  = int(faceW * 0.75)
        cx     = int(top.x * w)
        cy     = int(top.y * h)
        img    = overlayPNG(img, filters["Hat"],
                            cx - faceW//2, cy - faceH, faceW, faceH)

    elif filterName == "Glasses":
        le  = lm[LEFT_EYE]
        re  = lm[RIGHT_EYE]
        eyeW = int(abs(re.x - le.x) * w * 1.6)
        eyeH = int(eyeW * 0.35)
        cx  = int((le.x + re.x) / 2 * w)
        cy  = int((le.y + re.y) / 2 * h)
        img = overlayPNG(img, filters["Glasses"],
                         cx - eyeW//2, cy - eyeH//2, eyeW, eyeH)

    elif filterName == "Sunglasses":
        le  = lm[LEFT_EYE]
        re  = lm[RIGHT_EYE]
        eyeW = int(abs(re.x - le.x) * w * 1.6)
        eyeH = int(eyeW * 0.35)
        cx  = int((le.x + re.x) / 2 * w)
        cy  = int((le.y + re.y) / 2 * h)
        img = overlayPNG(img, filters["Sunglasses"],
                         cx - eyeW//2, cy - eyeH//2, eyeW, eyeH)

    elif filterName == "Crown":
        top   = lm[FOREHEAD]
        left  = lm[LEFT_TEMPLE]
        right = lm[RIGHT_TEMPLE]
        faceW = int(abs(right.x - left.x) * w * 1.3)
        faceH = int(faceW * 0.6)
        cx    = int(top.x * w)
        cy    = int(top.y * h)
        img   = overlayPNG(img, filters["Crown"],
                           cx - faceW//2, cy - faceH, faceW, faceH)

    elif filterName == "Mustache":
        nose  = lm[NOSE_BOTTOM]
        ml    = lm[MOUTH_LEFT]
        mr    = lm[MOUTH_RIGHT]
        mouthW = int(abs(mr.x - ml.x) * w * 1.8)
        mouthH = int(mouthW * 0.4)
        cx    = int(nose.x * w)
        cy    = int(nose.y * h) + 10
        img   = overlayPNG(img, filters["Mustache"],
                           cx - mouthW//2, cy, mouthW, mouthH)

    return img


print("Face Filter App Started!")
print("Controls:")
print("  A / D     = previous / next filter")
print("  A         = show ALL filters together")
print("  S         = screenshot")
print("  Q         = quit")

with FaceLandmarker.create_from_options(options) as detector:
    while True:
        success, img = cap.read()
        if not success:
            break

        img  = cv2.flip(img, 1)
        h, w, _ = img.shape

        imgRGB   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result   = detector.detect(mp_image)

        faceCount = 0

        if result.face_landmarks:
            faceCount = len(result.face_landmarks)
            for landmarks in result.face_landmarks:
                if showAll:
                    for name in filterNames:
                        img = applyFilter(img, landmarks, name, w, h)
                else:
                    img = applyFilter(img, landmarks,
                                      filterNames[currentFilter], w, h)

        # ── Top HUD ──
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # FPS
        cTime = time.time()
        fps   = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (0, 255, 100), 2)

        # Filter name
        filterLabel = "ALL FILTERS" if showAll else filterNames[currentFilter]
        cv2.putText(img, filterLabel, (w//2 - 100, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 200, 255), 3)

        # Face count
        cv2.putText(img, f"Faces: {faceCount}", (w - 200, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 200, 0), 2)

        # ── Filter selector bar ──
        barY = h - 60
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, barY - 10), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay2, 0.75, img, 0.25, 0, img)

        spacing = w // (len(filterNames) + 1)
        for i, name in enumerate(filterNames):
            x = spacing * (i + 1)
            color  = (0, 255, 100) if (i == currentFilter and not showAll) else (150, 150, 150)
            cv2.putText(img, name, (x - 40, h - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
            if i == currentFilter and not showAll:
                cv2.line(img, (x - 40, h - 10), (x + 40, h - 10), color, 2)

        # Controls
        cv2.putText(img, "A/D=Switch  SPACE=All  S=Screenshot  Q=Quit",
                    (20, barY - 15),
                    cv2.FONT_HERSHEY_PLAIN, 1.3, (180, 180, 180), 1)

        cv2.imshow("Snapchat Face Filter", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            currentFilter = (currentFilter + 1) % len(filterNames)
            showAll = False
        elif key == ord('a'):
            currentFilter = (currentFilter - 1) % len(filterNames)
            showAll = False
        elif key == ord(' '):
            showAll = not showAll
        elif key == ord('s'):
            fname = f"filter_screenshot_{screenshotCount}.png"
            cv2.imwrite(fname, img)
            screenshotCount += 1
            print(f"Screenshot saved: {fname}")

cap.release()
cv2.destroyAllWindows()
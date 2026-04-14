import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
import time

# Download model once
if not os.path.exists("hand_landmarker.task"):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Done!")

# Colors
colors = {
    'Red':    (0, 0, 255),
    'Green':  (0, 255, 0),
    'Blue':   (255, 0, 0),
    'Yellow': (0, 255, 255),
    'White':  (255, 255, 255),
}
colorNames = list(colors.keys())
colorValues = list(colors.values())

drawColor = colorValues[0]
brushThickness = 15
eraserThickness = 50

# Canvas to draw on
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgCanvas = np.zeros((480, 640, 3), np.uint8)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

tipIds = [4, 8, 12, 16, 20]

xp, yp = 0, 0
pTime = 0


def getFingersUp(lmList):
    fingers = []
    # Thumb
    if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def drawHeader(img, drawColor, brushThickness):
    # Draw toolbar background
    cv2.rectangle(img, (0, 0), (640, 80), (50, 50, 50), cv2.FILLED)

    # Draw color buttons
    for i, (name, color) in enumerate(colors.items()):
        x = 10 + i * 120
        cv2.rectangle(img, (x, 10), (x + 100, 70), color, cv2.FILLED)
        cv2.rectangle(img, (x, 10), (x + 100, 70), (255, 255, 255), 2)
        cv2.putText(img, name, (x + 5, 65),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        # Highlight selected color
        if color == drawColor:
            cv2.rectangle(img, (x - 3, 7), (x + 103, 73), (255, 255, 255), 3)

    # Eraser button
    cv2.rectangle(img, (610, 10), (635, 70), (100, 100, 100), cv2.FILLED)
    cv2.putText(img, "E", (613, 55),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Brush size indicator
    cv2.putText(img, f'Size:{brushThickness}', (540, 55),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    return img


print("AI Virtual Painter Started!")
print("Gestures:")
print("  Index finger only = DRAW")
print("  Index + Middle up = SELECT / MOVE (no draw)")
print("  Point at top bar = change color / eraser")
print("  All fingers up = CLEAR canvas")
print("  Pinky only up = INCREASE brush size")
print("  Thumb only up = DECREASE brush size")
print("Press Q to quit")

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = detector.detect(mp_image)

        lmList = []

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            points = []
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                lmList.append([cx, cy])

            # Draw hand skeleton
            for s, e in CONNECTIONS:
                cv2.line(img, points[s], points[e], (0, 255, 0), 1)
            for cx, cy in points:
                cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)

        if lmList:
            fingers = getFingersUp(lmList)

            x1, y1 = lmList[8]   # Index tip
            x2, y2 = lmList[12]  # Middle tip

            # ── CLEAR canvas (all 5 fingers up) ──
            if fingers == [1, 1, 1, 1, 1]:
                imgCanvas = np.zeros((480, 640, 3), np.uint8)
                xp, yp = 0, 0
                cv2.putText(img, "CLEARED!", (220, 250),
                            cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4)

            # ── INCREASE brush size (pinky only) ──
            elif fingers == [0, 0, 0, 0, 1]:
                brushThickness = min(brushThickness + 1, 60)
                xp, yp = 0, 0

            # ── DECREASE brush size (thumb only) ──
            elif fingers == [1, 0, 0, 0, 0]:
                brushThickness = max(brushThickness - 1, 5)
                xp, yp = 0, 0

            # ── SELECTION MODE (index + middle up) ──
            elif fingers[1] == 1 and fingers[2] == 1:
                xp, yp = 0, 0
                cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
                cv2.putText(img, "SELECT", (20, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                # Check if pointing at toolbar
                if y1 < 80:
                    # Color buttons
                    for i, color in enumerate(colorValues):
                        x = 10 + i * 120
                        if x < x1 < x + 100:
                            drawColor = color

                    # Eraser
                    if x1 > 610:
                        drawColor = (0, 0, 0)

            # ── DRAW MODE (index finger only) ──
            elif fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(img, (x1, y1), brushThickness // 2,
                           drawColor, cv2.FILLED)
                cv2.putText(img, "DRAW", (20, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, drawColor, 2)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness

                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                xp, yp = x1, y1

            else:
                xp, yp = 0, 0

        # Merge canvas with camera feed
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        # Draw toolbar on top
        img = drawHeader(img, drawColor, brushThickness)

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (550, 110),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("AI Virtual Painter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
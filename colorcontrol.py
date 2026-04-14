import cv2
import numpy as np
import pyautogui
import time
import os
import urllib.request
import mediapipe as mp

# Download model once
if not os.path.exists("hand_landmarker.task"):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Done!")

# Screen size
screenW, screenH = pyautogui.size()
pyautogui.FAILSAFE = False

# Camera size
wCam, hCam = 640, 480

# Smoothening
smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Frame reduction (border area to ignore)
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

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

pTime = 0
clickCooldown = 0

print("Hand Mouse Control Started!")
print("Index finger UP = Move mouse")
print("Index + Middle UP = Left Click")
print("Thumb + Index close = Right Click")
print("All fingers UP = Scroll mode")
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

            # Draw hand
            points = []
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                lmList.append([cx, cy])

            for s, e in CONNECTIONS:
                cv2.line(img, points[s], points[e], (0, 255, 0), 2)
            for cx, cy in points:
                cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        # Draw active area border
        cv2.rectangle(img, (frameR, frameR),
                      (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if lmList:
            fingers = getFingersUp(lmList)

            # Index finger tip position
            x1, y1 = lmList[8]
            # Middle finger tip
            x2, y2 = lmList[12]
            # Thumb tip
            xt, yt = lmList[4]

            # ── MODE 1: Move mouse (only index finger up) ──
            if fingers[1] == 1 and fingers[2] == 0:
                # Convert coords to screen size
                x3 = np.interp(x1, [frameR, wCam - frameR], [0, screenW])
                y3 = np.interp(y1, [frameR, hCam - frameR], [0, screenH])

                # Smoothen movement
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY

                cv2.circle(img, (x1, y1), 12, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "MOVE", (20, 100),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # ── MODE 2: Left Click (index + middle up) ──
            elif fingers[1] == 1 and fingers[2] == 1:
                dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                if dist < 35 and clickCooldown == 0:
                    pyautogui.click()
                    clickCooldown = 15
                    cv2.circle(img, ((x1+x2)//2, (y1+y2)//2),
                               12, (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "LEFT CLICK!", (20, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "PINCH TO CLICK", (20, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # ── MODE 3: Right Click (thumb + index close) ──
            elif fingers[0] == 1 and fingers[1] == 0:
                dist = ((xt - x1)**2 + (yt - y1)**2)**0.5
                if dist < 40 and clickCooldown == 0:
                    pyautogui.rightClick()
                    clickCooldown = 20
                    cv2.putText(img, "RIGHT CLICK!", (20, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # ── MODE 4: Scroll (all fingers up) ──
            elif fingers == [0, 1, 1, 1, 1]:
                prev_y = y1
                if y1 < hCam // 2:
                    pyautogui.scroll(3)
                    cv2.putText(img, "SCROLL UP", (20, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                else:
                    pyautogui.scroll(-3)
                    cv2.putText(img, "SCROLL DOWN", (20, 100),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # Cooldown timer
        if clickCooldown > 0:
            clickCooldown -= 1

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Hand Mouse Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
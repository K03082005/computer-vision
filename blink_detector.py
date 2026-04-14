import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request

# Download face landmarker model
if not os.path.exists("face_landmarker.task"):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker.task"
    )
    print("Done!")

# MediaPipe Face setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ── Eye landmark indices for MediaPipe Face Mesh ──
# Left eye
LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_TOP    = [386, 387, 388]
LEFT_EYE_BOTTOM = [374, 373, 390]
LEFT_EYE_LEFT   = [263]
LEFT_EYE_RIGHT  = [362]

# Right eye
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_TOP    = [159, 160, 158]
RIGHT_EYE_BOTTOM = [145, 144, 163]
RIGHT_EYE_LEFT   = [133]
RIGHT_EYE_RIGHT  = [33]

# EAR threshold and frame count
EAR_THRESHOLD   = 0.22
BLINK_FRAMES    = 2   # frames eye must be closed to count as blink
DROWSY_BLINKS   = 15  # blinks per minute to trigger drowsy
DROWSY_EAR_TIME = 2.0 # seconds eyes closed = drowsy alert

# Stats
blinkCount     = 0
blinkPerMin    = 0
framesClosed   = 0
eyeClosedStart = 0
isDrowsy       = False
isAlert        = False
alertStart     = 0
blinkTimes     = []
pTime          = 0
faceFound      = False

# Graph data
earHistory     = []
MAX_HISTORY    = 150


def eyeAspectRatio(landmarks, top_ids, bottom_ids, left_id, right_id, w, h):
    top    = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in top_ids], axis=0)
    bottom = np.mean([[landmarks[i].x * w, landmarks[i].y * h] for i in bottom_ids], axis=0)
    left   = np.array([landmarks[left_id[0]].x * w,  landmarks[left_id[0]].y * h])
    right  = np.array([landmarks[right_id[0]].x * w, landmarks[right_id[0]].y * h])

    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    ear = vertical / horizontal if horizontal != 0 else 0
    return ear


def drawEye(img, landmarks, eye_ids, color, w, h):
    pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]
                    for i in eye_ids], np.int32)
    cv2.polylines(img, [pts], True, color, 1)


def drawRoundedRect(img, x1, y1, x2, y2, color, alpha=0.7, radius=10):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def drawEARGraph(img, earHistory, threshold, x, y, gw, gh):
    # Background
    cv2.rectangle(img, (x, y), (x + gw, y + gh), (30, 30, 30), cv2.FILLED)
    cv2.rectangle(img, (x, y), (x + gw, y + gh), (100, 100, 100), 1)

    # Threshold line
    ty = int(y + gh - (threshold / 0.5) * gh)
    cv2.line(img, (x, ty), (x + gw, ty), (0, 100, 255), 1)
    cv2.putText(img, f"Threshold", (x + 2, ty - 3),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 255), 1)

    # EAR line
    if len(earHistory) > 1:
        for i in range(1, len(earHistory)):
            x1 = x + int((i - 1) / MAX_HISTORY * gw)
            x2 = x + int(i / MAX_HISTORY * gw)
            y1 = int(y + gh - (min(earHistory[i-1], 0.5) / 0.5) * gh)
            y2 = int(y + gh - (min(earHistory[i],   0.5) / 0.5) * gh)
            color = (0, 255, 100) if earHistory[i] > threshold else (0, 0, 255)
            cv2.line(img, (x1, y1), (x2, y2), color, 2)

    cv2.putText(img, "EAR Graph", (x + gw//2 - 30, y + gh + 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)


print("Eye Blink Drowsiness Detector Started!")
print("Sit in front of camera with good lighting")
print("Press Q to quit | Press R to reset stats")

with FaceLandmarker.create_from_options(options) as detector:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = detector.detect(mp_image)

        ear = 0
        faceFound = False

        if result.face_landmarks:
            faceFound = True
            landmarks = result.face_landmarks[0]

            # Calculate EAR for both eyes
            leftEAR  = eyeAspectRatio(landmarks, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,
                                       LEFT_EYE_LEFT,  LEFT_EYE_RIGHT, w, h)
            rightEAR = eyeAspectRatio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                                       RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT, w, h)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eyes
            eyeColor = (0, 255, 100) if ear > EAR_THRESHOLD else (0, 0, 255)
            drawEye(img, landmarks, LEFT_EYE,  eyeColor, w, h)
            drawEye(img, landmarks, RIGHT_EYE, eyeColor, w, h)

            # Draw all face dots lightly
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 1, (80, 80, 80), cv2.FILLED)

            # ── Blink Detection ──
            if ear < EAR_THRESHOLD:
                framesClosed += 1
                if eyeClosedStart == 0:
                    eyeClosedStart = time.time()
            else:
                if framesClosed >= BLINK_FRAMES:
                    blinkCount += 1
                    blinkTimes.append(time.time())
                framesClosed  = 0
                eyeClosedStart = 0
                isDrowsy      = False

            # ── Drowsiness from prolonged eye closure ──
            if eyeClosedStart > 0:
                closedDuration = time.time() - eyeClosedStart
                if closedDuration > DROWSY_EAR_TIME:
                    isDrowsy  = True
                    isAlert   = True
                    alertStart = time.time()

            # ── Blinks per minute ──
            now = time.time()
            blinkTimes = [t for t in blinkTimes if now - t < 60]
            blinkPerMin = len(blinkTimes)

            # High blink rate = drowsy
            if blinkPerMin > DROWSY_BLINKS:
                isDrowsy  = True
                isAlert   = True
                alertStart = time.time()

        # EAR history for graph
        earHistory.append(ear)
        if len(earHistory) > MAX_HISTORY:
            earHistory.pop(0)

        # ── DROWSY ALERT ──
        if isAlert:
            if time.time() - alertStart < 3:
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), cv2.FILLED)
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                cv2.putText(img, "DROWSY ALERT!", (w//2 - 200, h//2 - 20),
                            cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 6)
                cv2.putText(img, "WAKE UP!", (w//2 - 110, h//2 + 60),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
            else:
                isAlert  = False
                isDrowsy = False

        # ── Top HUD ──
        drawRoundedRect(img, 0, 0, w, 75, (20, 20, 20), 0.85)

        cTime = time.time()
        fps   = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 55),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 100), 3)

        statusText  = "DROWSY!" if isDrowsy else ("ALERT" if faceFound else "NO FACE")
        statusColor = (0, 0, 255) if isDrowsy else ((0, 255, 100) if faceFound else (0, 200, 255))
        cv2.putText(img, statusText, (w//2 - 80, 55),
                    cv2.FONT_HERSHEY_PLAIN, 3, statusColor, 3)

        cv2.putText(img, f"EAR: {ear:.2f}", (w - 220, 55),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 100) if ear > EAR_THRESHOLD else (0, 0, 255), 3)

        # ── Stats Panel ──
        drawRoundedRect(img, 20, 90, 280, 320, (20, 20, 20), 0.75)

        cv2.putText(img, "STATS", (30, 120),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.line(img, (30, 128), (270, 128), (100, 100, 100), 1)

        cv2.putText(img, f"Blinks: {blinkCount}", (30, 160),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 255), 2)
        cv2.putText(img, f"Per min: {blinkPerMin}", (30, 195),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 200, 255), 2)

        # Eye closed duration bar
        closedDur = time.time() - eyeClosedStart if eyeClosedStart > 0 else 0
        cv2.putText(img, f"Closed: {closedDur:.1f}s", (30, 230),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 150, 255), 2)

        barW = 220
        prog = min(closedDur / DROWSY_EAR_TIME, 1.0)
        cv2.rectangle(img, (30, 240), (30 + barW, 258), (50, 50, 50), cv2.FILLED)
        barColor = (0, 255, 100) if prog < 0.6 else ((0, 200, 255) if prog < 1.0 else (0, 0, 255))
        cv2.rectangle(img, (30, 240), (30 + int(barW * prog), 258), barColor, cv2.FILLED)
        cv2.putText(img, "Drowsy threshold", (30, 278),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (150, 150, 150), 1)

        # ── EAR Graph ──
        drawEARGraph(img, earHistory, EAR_THRESHOLD,
                     w - 330, h - 160, 310, 130)

        # ── Instructions ──
        cv2.putText(img, "R = Reset  |  Q = Quit", (20, h - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 150, 150), 1)

        cv2.imshow("Eye Blink Drowsiness Detector", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            blinkCount     = 0
            blinkTimes     = []
            framesClosed   = 0
            eyeClosedStart = 0
            isDrowsy       = False
            isAlert        = False
            earHistory     = []
            print("Stats reset!")

print(f"\nSession Summary:")
print(f"  Total Blinks: {blinkCount}")
print(f"  Blinks/min at exit: {blinkPerMin}")

cap.release()
cv2.destroyAllWindows()
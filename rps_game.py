import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import urllib.request

# Download model once
if not os.path.exists("hand_landmarker.task"):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Done!")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
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

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Game state
playerScore = 0
aiScore = 0
ties = 0
gameState = "waiting"   # waiting, countdown, result
countdown = 3
countdownTimer = 0
resultTimer = 0
playerMove = ""
aiMove = ""
result = ""
lastGesture = ""
stableCount = 0
STABLE_FRAMES = 20  # frames gesture must be held


def getFingersUp(lmList):
    fingers = []
    # Thumb - check x axis
    if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    # 4 Fingers - check y axis
    for id in range(1, 5):
        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def getGesture(fingers):
    totalFingers = sum(fingers)
    # Rock = all fingers closed
    if totalFingers == 0:
        return "ROCK"
    # Paper = all fingers open
    elif totalFingers == 5:
        return "PAPER"
    # Scissors = index + middle only
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "SCISSORS"
    else:
        return "UNKNOWN"


def getWinner(player, ai):
    if player == ai:
        return "TIE"
    elif (player == "ROCK" and ai == "SCISSORS") or \
         (player == "SCISSORS" and ai == "PAPER") or \
         (player == "PAPER" and ai == "ROCK"):
        return "WIN"
    else:
        return "LOSE"


def drawGestureEmoji(img, gesture, x, y, size=80):
    emojis = {
        "ROCK":     "✊",
        "PAPER":    "🖐",
        "SCISSORS": "✌",
        "UNKNOWN":  "?"
    }
    shapes = {
        "ROCK": lambda: cv2.circle(img, (x, y), size // 2, (100, 100, 100), cv2.FILLED),
        "PAPER": lambda: cv2.rectangle(img, (x - size//2, y - size//2),
                                       (x + size//2, y + size//2), (200, 200, 50), cv2.FILLED),
        "SCISSORS": lambda: [cv2.line(img, (x, y), (x - size//2, y - size//2), (50, 200, 50), 4),
                              cv2.line(img, (x, y), (x + size//2, y - size//2), (50, 200, 50), 4)],
    }
    # Draw shape
    if gesture in shapes:
        shapes[gesture]()
    # Draw label
    cv2.putText(img, gesture, (x - size//2, y + size//2 + 25),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)


def drawRoundedRect(img, x1, y1, x2, y2, color, alpha=0.6):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


pTime = 0

print("Rock Paper Scissors Game Started!")
print("Show your gesture and HOLD it steady for 2 seconds to play!")
print("Press R to reset scores | Press Q to quit")

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result_mp = detector.detect(mp_image)

        lmList = []

        if result_mp.hand_landmarks:
            hand = result_mp.hand_landmarks[0]
            points = []
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                points.append((cx, cy))
                lmList.append([cx, cy])

            for s, e in CONNECTIONS:
                cv2.line(img, points[s], points[e], (0, 200, 0), 2)
            for cx, cy in points:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # ── Dark background panel top ──
        drawRoundedRect(img, 0, 0, w, 70, (20, 20, 20), 0.8)

        # ── Score display ──
        cv2.putText(img, f"YOU: {playerScore}", (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 100), 3)
        cv2.putText(img, f"AI: {aiScore}", (w - 180, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 100, 255), 3)
        cv2.putText(img, f"TIES: {ties}", (w//2 - 60, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 200), 2)

        # ── Game Logic ──
        if lmList:
            fingers = getFingersUp(lmList)
            gesture = getGesture(fingers)

            if gameState == "waiting":
                # Check gesture stability
                if gesture == lastGesture and gesture != "UNKNOWN":
                    stableCount += 1
                else:
                    stableCount = 0
                    lastGesture = gesture

                # Show current gesture
                cv2.putText(img, f"Gesture: {gesture}", (20, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                # Progress bar for stability
                if gesture != "UNKNOWN":
                    progress = int((stableCount / STABLE_FRAMES) * (w - 40))
                    cv2.rectangle(img, (20, 130), (w - 20, 150), (50, 50, 50), cv2.FILLED)
                    cv2.rectangle(img, (20, 130), (20 + progress, 150), (0, 255, 100), cv2.FILLED)
                    cv2.putText(img, "Hold steady...", (20, 175),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

                # Trigger game when stable
                if stableCount >= STABLE_FRAMES and gesture != "UNKNOWN":
                    playerMove = gesture
                    aiMove = random.choice(["ROCK", "PAPER", "SCISSORS"])
                    result = getWinner(playerMove, aiMove)

                    if result == "WIN":
                        playerScore += 1
                    elif result == "LOSE":
                        aiScore += 1
                    else:
                        ties += 1

                    gameState = "result"
                    resultTimer = time.time()
                    stableCount = 0

            elif gameState == "result":
                # Show result panel
                drawRoundedRect(img, 20, 80, w - 20, 420, (20, 20, 40), 0.85)

                # Player move
                cv2.putText(img, "YOU", (80, 120),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 100), 2)
                drawGestureEmoji(img, playerMove, 120, 220)

                # VS
                cv2.putText(img, "VS", (w//2 - 20, 230),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                # AI move
                cv2.putText(img, "AI", (w - 180, 120),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 255), 2)
                drawGestureEmoji(img, aiMove, w - 120, 220)

                # Result text
                if result == "WIN":
                    color = (0, 255, 100)
                    text = "YOU WIN!"
                elif result == "LOSE":
                    color = (0, 100, 255)
                    text = "AI WINS!"
                else:
                    color = (255, 255, 0)
                    text = "TIE!"

                cv2.putText(img, text, (w//2 - 80, 340),
                            cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

                # Countdown to next round
                elapsed = time.time() - resultTimer
                remaining = max(0, 3 - int(elapsed))
                cv2.putText(img, f"Next round in {remaining}...", (w//2 - 130, 400),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

                if elapsed > 3:
                    gameState = "waiting"
                    lastGesture = ""
                    stableCount = 0

        else:
            # No hand detected
            cv2.putText(img, "Show your hand!", (w//2 - 130, h//2),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            gameState = "waiting"
            stableCount = 0

        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (w - 110, h - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        # Instructions
        cv2.putText(img, "R=Reset Q=Quit", (20, h - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 150, 150), 1)

        cv2.imshow("Rock Paper Scissors vs AI", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            playerScore, aiScore, ties = 0, 0, 0
            gameState = "waiting"
            stableCount = 0

cap.release()
cv2.destroyAllWindows()
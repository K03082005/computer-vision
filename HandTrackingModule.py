import cv2
import mediapipe as mp
import time
import math
import os
import urllib.request

if not os.path.exists("hand_landmarker.task"):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task"
    )
    print("Done!")

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.lmList = []
        self.imgShape = (480, 640)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.detector = HandLandmarker.create_from_options(options)
        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        h, w, _ = img.shape
        self.imgShape = (h, w)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mp_image)

        if self.results.hand_landmarks:
            for hand in self.results.hand_landmarks:
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                if draw:
                    for s, e in CONNECTIONS:
                        cv2.line(img, points[s], points[e], (0, 255, 0), 2)
                    for cx, cy in points:
                        cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList = [], []
        bbox = []
        self.lmList = []
        h, w = self.imgShape

        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[handNo]
                for id, lm in enumerate(hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
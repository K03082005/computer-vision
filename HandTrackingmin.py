import cv2
import mediapipe as mp
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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,  # lower = detects more easily
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
pTime = 0

# Connection pairs for drawing lines manually
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17)            # palm
]

with HandLandmarker.create_from_options(options) as detector:
    while True:
        success, img = cap.read()
        if not success:
            print("Camera error! Try VideoCapture(1)")
            break

        h, w, _ = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            
import cv2
import time
from ultralytics import YOLO
import numpy as np

# Load YOLO model (downloads automatically first time ~6MB)
print("Loading YOLO model...")
model = YOLO("yolov8n.pt")  # nano = fastest, most accurate tradeoff
print("Model loaded!")

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# 80 COCO class colors - one per class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)

pTime = 0
confidence = 0.4  # detection threshold
paused = False

# Stats
frameCount = 0
totalDetections = 0

print("Object Detector Started!")
print("Controls:")
print("  + / -   = increase / decrease confidence threshold")
print("  P       = pause / resume")
print("  S       = screenshot")
print("  Q       = quit")

while True:
    if not paused:
        success, img = cap.read()
        if not success:
            break

        frameCount += 1

        # Run YOLO detection
        results = model(img, stream=True, conf=confidence, verbose=False)

        detectionCount = 0
        classesDetected = {}

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence
                conf = float(box.conf[0])

                # Class
                cls = int(box.cls[0])
                className = model.names[cls]

                # Track detections
                detectionCount += 1
                if className not in classesDetected:
                    classesDetected[className] = 0
                classesDetected[className] += 1

                # Color per class
                color = tuple(map(int, COLORS[cls % 80]))

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Label background
                label = f"{className} {conf:.0%}"
                labelSize = cv2.getTextSize(label,
                                            cv2.FONT_HERSHEY_PLAIN, 1.5, 2)[0]
                cv2.rectangle(img,
                              (x1, y1 - labelSize[1] - 10),
                              (x1 + labelSize[0] + 10, y1),
                              color, cv2.FILLED)

                # Label text
                cv2.putText(img, label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1.5,
                            (255, 255, 255), 2)

                # Confidence bar inside box
                barW = x2 - x1
                cv2.rectangle(img, (x1, y2 - 8), (x2, y2), (50, 50, 50), cv2.FILLED)
                cv2.rectangle(img, (x1, y2 - 8),
                              (x1 + int(barW * conf), y2),
                              color, cv2.FILLED)

        totalDetections += detectionCount

    # ── HUD Panel ──
    h, w, _ = img.shape
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(overlay, (0, h - 120), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime

    # Top bar
    cv2.putText(img, f"FPS: {int(fps)}", (20, 55),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 100), 3)
    cv2.putText(img, f"Objects: {detectionCount}", (200, 55),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 200, 255), 3)
    cv2.putText(img, f"Conf: {int(confidence * 100)}%", (500, 55),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 200, 0), 3)

    if paused:
        cv2.putText(img, "PAUSED", (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    # Bottom bar - show detected classes
    cv2.putText(img, "Detected:", (20, h - 90),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

    xOffset = 130
    for name, count in list(classesDetected.items())[:8]:
        cls = list(model.names.values()).index(name) if name in model.names.values() else 0
        color = tuple(map(int, COLORS[cls % 80]))
        text = f"{name}({count})"
        cv2.putText(img, text, (xOffset, h - 90),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
        xOffset += len(text) * 12 + 10

    # Controls reminder
    cv2.putText(img, "+/- Confidence  |  P Pause  |  S Screenshot  |  Q Quit",
                (20, h - 15),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 150, 150), 1)

    cv2.imshow("YOLO Object Detector", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused
    elif key == ord('+') or key == ord('='):
        confidence = min(confidence + 0.05, 0.95)
        print(f"Confidence: {confidence:.0%}")
    elif key == ord('-'):
        confidence = max(confidence - 0.05, 0.05)
        print(f"Confidence: {confidence:.0%}")
    elif key == ord('s'):
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, img)
        print(f"Screenshot saved: {filename}")

print(f"\nSession Stats:")
print(f"  Total Frames: {frameCount}")
print(f"  Total Detections: {totalDetections}")

cap.release()
cv2.destroyAllWindows()
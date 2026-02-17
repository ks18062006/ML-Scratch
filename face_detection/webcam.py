import cv2 as cv
import time
import os

MODEL = "models/face_detection_yunet_2023mar.onnx"

if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Model not found: {MODEL}")

# Create detector
detector = cv.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:
        for f in faces:
            x, y, bw, bh = map(int, f[:4])
            score = float(f[14])

            cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv.putText(frame, f"{score:.2f}", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS calculation
    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now

    cv.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("Face Cam (Press q to exit)", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

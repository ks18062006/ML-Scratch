import cv2 as cv

MODEL = "models/face_detection_yunet_2023mar.onnx"

img = cv.imread("test.jpg")
if img is None:
    raise FileNotFoundError("This error")

h, w = img.shape[:2]

detector = cv.FaceDetectorYN.create(MODEL, "", (320, 320), 0.9, 0.3, 5000)
detector.setInputSize((w, h))

_, faces = detector.detect(img)
faces = faces if faces is not None else []

for f in faces:
    x, y, bw, bh = map(int, f[:4])
    score = float(f[14])
    cv.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv.putText(img, f"{score:.2f}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

print("Faces detected:", len(faces))
cv.imshow("YuNet", img)
cv.waitKey(0)
cv.destroyAllWindows()

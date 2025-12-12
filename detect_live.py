from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Confidence mínimo 0.60 para evitar falsos positivos
    results = model(frame, conf=0.60)

    annotated = results[0].plot()

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]

        print(f"{label}: {conf*100:.1f}%")

    cv2.imshow("Detector de Jabalíes", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

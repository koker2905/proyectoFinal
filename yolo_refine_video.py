from ultralytics import YOLO
import cv2
import sys
import os

video_path = sys.argv[1]

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "video_yolo.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

print("🎥 Procesando video con YOLO...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, classes=[0])

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("✅ Video con cajas YOLO generado: video_yolo.mp4")
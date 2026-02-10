from ultralytics import YOLO
import cv2
import sys
import subprocess

img_path = sys.argv[1]

model = YOLO("yolov8n.pt")

results = model(img_path, conf=0.4, classes=[0])

img = cv2.imread(img_path)
best_box = None
best_conf = 0

for r in results:
    for box in r.boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box.xyxy[0]

if best_box is not None:
    x1, y1, x2, y2 = map(int, best_box)
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.imwrite("yolo_refined.jpg", img)

    crop = img[y1:y2, x1:x2]
    cv2.imwrite("person_crop.jpg", crop)

    # 👉 LLAMAR POSTURA
    subprocess.run(["python3", "pose_detect.py"])
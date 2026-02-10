import cv2
import os

OUTPUT_DIR = "hog_dataset/positives"
splits = ["train", "valid", "test"]
IMG_SIZE = (64, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)

counter = 0
total_boxes = 0
kept_boxes = 0

for split in splits:
    IMAGES_DIR = f"dataset/{split}/images"
    LABELS_DIR = f"dataset/{split}/labels"

    if not os.path.exists(IMAGES_DIR):
        continue

    for img_name in os.listdir(IMAGES_DIR):
        if not img_name.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(
            LABELS_DIR,
            img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img, _ = img.shape

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls, x, y, bw, bh = map(float, parts[:5])
            total_boxes += 1

            # YOLO → píxeles
            x1 = int((x - bw / 2) * w_img)
            y1 = int((y - bh / 2) * h_img)
            x2 = int((x + bw / 2) * w_img)
            y2 = int((y + bh / 2) * h_img)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img - 1, x2)
            y2 = min(h_img - 1, y2)

            w = x2 - x1
            h = y2 - y1

            # 🔧 FILTROS MÁS REALISTAS PARA INRIA
            if h < 80 or w < 30:
                continue

            ratio = h / w
            if ratio < 1.5 or ratio > 4.0:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, IMG_SIZE)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, f"pos_{counter}.png"),
                crop
            )

            counter += 1
            kept_boxes += 1

print("======== RESUMEN ========")
print(f"Total bounding boxes leídas: {total_boxes}")
print(f"Patches positivos generados: {kept_boxes}")
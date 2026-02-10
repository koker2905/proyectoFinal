import cv2
import os
import albumentations as A

SRC = "hog_dataset/negatives"
DST = "hog_dataset/negatives"
os.makedirs(DST, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussianBlur(p=0.2),
    A.MotionBlur(p=0.2),
    A.Rotate(limit=8, p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.3)
])

counter = len(os.listdir(DST))

for img_name in os.listdir(SRC):
    if not img_name.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(SRC, img_name))
    if img is None:
        continue

    for i in range(6):  # 🔥 6 variantes
        aug = transform(image=img)["image"]
        aug = cv2.resize(aug, (64, 128))
        cv2.imwrite(os.path.join(DST, f"neg_aug_{counter}.png"), aug)
        counter += 1

print("✅ Augmentation NEGATIVOS listo")
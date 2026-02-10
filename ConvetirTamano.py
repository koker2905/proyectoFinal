import cv2, os

SRC = "Negativos"
DST = "hog_dataset/negatives"
os.makedirs(DST, exist_ok=True)

i = 0
for img_name in os.listdir(SRC):
    img = cv2.imread(os.path.join(SRC, img_name))
    img = cv2.resize(img, (64, 128))
    cv2.imwrite(os.path.join(DST, f"neg_{i}.png"), img)
    i += 1
import cv2
import mediapipe as mp
import sys

input_image = sys.argv[1]
output_image = "pose_output.jpg"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

image = cv2.imread(input_image)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(rgb)

if results.pose_landmarks:
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

cv2.imwrite(output_image, image)
print("✅ Postura detectada")
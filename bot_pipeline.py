from telegram import Bot
import subprocess
import os
import sys
import psutil
import time

# ===== CONFIG =====
TOKEN = "8448132878:AAG4T5i4874lrRpekMVQraYjquuww-uJf-w"
CHAT_ID = 737007959
bot = Bot(token=TOKEN)

if len(sys.argv) < 2:
    print("❌ No se recibió el video")
    sys.exit(1)

video_path = sys.argv[1]

print("🤖 Bot iniciado")
print(f"📹 Video recibido: {video_path}")

process = psutil.Process(os.getpid())

# ===== IMAGEN ORIGINAL =====
if os.path.exists("hog_frame.jpg"):
    bot.send_photo(
        chat_id=CHAT_ID,
        photo=open("hog_frame.jpg", "rb"),
        caption="📸 Imagen original detectada"
    )

# ===== YOLO =====
start = time.time()
subprocess.run(["python3", "yolo_refine_video.py", video_path])
yolo_time = time.time() - start

# ===== POSTURA =====
start = time.time()
subprocess.run(["python3", "pose_video.py", "video_yolo.mp4"])
pose_time = time.time() - start

# ===== MEMORIA =====
mem_mb = process.memory_info().rss / 1024 / 1024

# ===== ENVIOS =====
if os.path.exists("video_pose.mp4"):
    bot.send_video(
        chat_id=CHAT_ID,
        video=open("video_pose.mp4", "rb"),
        caption="🎥 Video con detección de postura humana"
    )

# ===== METRICAS =====
bot.send_message(
    chat_id=CHAT_ID,
    text=(
        "📊 Métricas del sistema:\n"
        f"- Tiempo YOLO: {yolo_time:.2f} s\n"
        f"- Tiempo Postura: {pose_time:.2f} s\n"
        f"- Memoria bot: {mem_mb:.2f} MB\n"
        "- Puntos postura: 33\n"
    )
)

print("✅ Proceso completo enviado al usuario")
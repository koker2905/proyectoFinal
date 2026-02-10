from telegram import Bot
import os

# 🔑 DATOS DEL BOT
TOKEN = "8448132878:AAG4T5i4874lrRpekMVQraYjquuww-uJf-w"
CHAT_ID = 737007959   # 👈 tu chat id

bot = Bot(token=TOKEN)

def send_results():
    print("📤 Enviando resultados por Telegram...")

    if os.path.exists("hog_frame.jpg"):
        bot.send_photo(
            chat_id=CHAT_ID,
            photo=open("hog_frame.jpg", "rb"),
            caption="📸 Imagen original detectada"
        )

    if os.path.exists("video_yolo.mp4"):
        bot.send_video(
            chat_id=CHAT_ID,
            video=open("video_yolo.mp4", "rb"),
            caption="🎥 Video con detección YOLO"
        )

    print("✅ Archivos enviados")

if __name__ == "__main__":
    send_results()
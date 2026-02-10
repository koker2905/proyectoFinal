from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/process", methods=["POST"])
def process_video():
    data = request.json
    video_path = data.get("video_path")

    if not video_path or not os.path.exists(video_path):
        return jsonify({"status": "error", "msg": "Video no encontrado"}), 400

    # 🔥 Ejecuta el pipeline del bot
    subprocess.run(["python3", "bot_pipeline.py", video_path])

    return jsonify({
        "status": "ok",
        "msg": "Video recibido y procesado"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
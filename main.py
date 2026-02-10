import subprocess
import os

print("🚀 Iniciando sistema híbrido HOG + YOLO + Postura")

# Verificar binario
if not os.path.exists("./test_hog_webcam"):
    print("❌ No existe el ejecutable test_hog_webcam")
    exit(1)

# Ejecutar detector clásico (C++)
subprocess.run(["./test_hog_webcam"])
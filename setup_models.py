"""Télécharge les modèles ONNX YuNet (détection) et SFace (reconnaissance)."""

import os
import urllib.request

MODELS_DIR = "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}

os.makedirs(MODELS_DIR, exist_ok=True)

for filename, url in MODELS.items():
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        print(f"[OK] {filename} déjà présent")
        continue
    print(f"[DOWNLOAD] {filename} ...")
    urllib.request.urlretrieve(url, path)
    size_kb = os.path.getsize(path) // 1024
    print(f"[OK] {filename} ({size_kb} Ko)")

print("[DONE] Modèles prêts.")

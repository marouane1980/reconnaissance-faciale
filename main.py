"""
Reconnaissance Faciale en Temps Réel
-------------------------------------
Utilise OpenCV pour la capture vidéo et face_recognition pour l'identification.

Commandes clavier :
  S  — sauvegarder le frame courant dans captured_faces/
  Q  — quitter
"""

import os
import time
import cv2
import numpy as np

from face_recognizer import FaceRecognizer

KNOWN_FACES_DIR = "known_faces"
CAPTURED_DIR = "captured_faces"
CAMERA_INDEX = 0
PROCESS_EVERY_N_FRAMES = 3  # traiter 1 frame sur N pour la performance


def draw_results(frame, results):
    for (top, right, bottom, left, name) in results:
        color = (0, 200, 0) if name != "Inconnu" else (0, 0, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        label_bg_top = bottom if bottom + 30 < frame.shape[0] else top - 30
        cv2.rectangle(frame, (left, label_bg_top), (right, label_bg_top + 25), color, cv2.FILLED)
        cv2.putText(
            frame, name, (left + 4, label_bg_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )


def main():
    os.makedirs(CAPTURED_DIR, exist_ok=True)

    recognizer = FaceRecognizer(KNOWN_FACES_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        return

    print("[INFO] Démarrage — appuyez sur S pour capturer, Q pour quitter.")

    frame_count = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy()

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            raw = recognizer.recognize(rgb_small)
            last_results = [(t * 2, r * 2, b * 2, l * 2, n) for (t, r, b, l, n) in raw]

        draw_results(display, last_results)

        face_count = len(last_results)
        cv2.putText(
            display,
            f"Visages: {face_count}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2
        )
        cv2.imshow("Reconnaissance Faciale — Q: quitter | S: capturer", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(CAPTURED_DIR, f"capture_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] Image sauvegardée: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()

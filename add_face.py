"""
Outil d'enregistrement d'un nouveau visage connu.

Usage:
    python add_face.py --name "Prénom Nom" --source webcam
    python add_face.py --name "Prénom Nom" --source image.jpg
"""

import argparse
import os
import cv2


KNOWN_FACES_DIR = "known_faces"


def capture_from_webcam(name, camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("[ERREUR] Caméra inaccessible.")
        return

    print("[INFO] Cadrez votre visage et appuyez sur ESPACE pour capturer, Q pour annuler.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Enregistrement — ESPACE: capturer | Q: annuler", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            filename = name.lower().replace(" ", "_") + ".jpg"
            path = os.path.join(KNOWN_FACES_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"[INFO] Visage enregistré: {path}")
            break
        elif key == ord("q"):
            print("[INFO] Annulé.")
            break

    cap.release()
    cv2.destroyAllWindows()


def copy_from_image(name, image_path):
    if not os.path.isfile(image_path):
        print(f"[ERREUR] Fichier introuvable: {image_path}")
        return
    frame = cv2.imread(image_path)
    filename = name.lower().replace(" ", "_") + ".jpg"
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, frame)
    print(f"[INFO] Visage enregistré: {path}")


def main():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    parser = argparse.ArgumentParser(description="Ajouter un visage connu.")
    parser.add_argument("--name", required=True, help="Nom de la personne")
    parser.add_argument(
        "--source", default="webcam",
        help="'webcam' ou chemin vers une image (défaut: webcam)"
    )
    args = parser.parse_args()

    if args.source == "webcam":
        capture_from_webcam(args.name)
    else:
        copy_from_image(args.name, args.source)


if __name__ == "__main__":
    main()

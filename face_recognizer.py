import os
import cv2
import numpy as np


class FaceRecognizer:
    """Reconnaissance faciale via OpenCV LBPH (sans dlib)."""

    def __init__(self, known_faces_dir="known_faces", threshold=80):
        self.threshold = threshold
        self.known_names = []
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.trained = False
        self.load_known_faces(known_faces_dir)

    def _detect_face(self, gray):
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return gray[y:y + h, x:x + w]

    def load_known_faces(self, directory):
        if not os.path.isdir(directory):
            return
        samples, labels = [], []
        names = []
        for filename in os.listdir(directory):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(directory, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face = self._detect_face(img)
            if face is None:
                continue
            face = cv2.resize(face, (200, 200))
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            if name not in names:
                names.append(name)
            label = names.index(name)
            samples.append(face)
            labels.append(label)

        if samples:
            self.known_names = names
            self.recognizer.train(samples, np.array(labels))
            self.trained = True
            print(f"[INFO] {len(names)} visage(s) chargé(s): {names}")
        else:
            print("[INFO] Aucun visage connu — mode détection uniquement.")

    def recognize(self, frame_bgr):
        """Retourne liste de (top, right, bottom, left, name) pour chaque visage."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        results = []
        for (x, y, w, h) in faces:
            name = "Inconnu"
            if self.trained:
                face_roi = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                label, confidence = self.recognizer.predict(face_roi)
                if confidence < self.threshold:
                    name = self.known_names[label]
            results.append((y, x + w, y + h, x, name))
        return results

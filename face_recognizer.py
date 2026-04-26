import os
import cv2
import numpy as np


class FaceRecognizer:
    """Reconnaissance faciale via OpenCV LBPH avec augmentation de données."""

    def __init__(self, known_faces_dir="known_faces", threshold=65):
        self.threshold = threshold
        self.known_names = []
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=16, grid_x=8, grid_y=8
        )
        self.trained = False
        self.load_known_faces(known_faces_dir)

    def _detect_face(self, gray):
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # Prendre le visage le plus grand détecté
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return gray[y:y + h, x:x + w]

    def _augment(self, face):
        """Génère des variantes pour améliorer la robustesse du modèle."""
        samples = [face]
        # Symétrie horizontale
        samples.append(cv2.flip(face, 1))
        # Variations de luminosité
        for gamma in (0.75, 1.3):
            lut = np.array([min(255, int(((i / 255.0) ** (1.0 / gamma)) * 255))
                            for i in range(256)], dtype=np.uint8)
            samples.append(cv2.LUT(face, lut))
        # Légères rotations (-10°, +10°)
        h, w = face.shape
        center = (w // 2, h // 2)
        for angle in (-10, 10):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(face, M, (w, h))
            samples.append(rotated)
        # Égalisation d'histogramme (améliore contraste)
        samples.append(cv2.equalizeHist(face))
        return samples

    def _preprocess(self, face):
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)
        return face

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
            face = self._preprocess(face)
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            if name not in names:
                names.append(name)
            label = names.index(name)
            for variant in self._augment(face):
                samples.append(variant)
                labels.append(label)

        if samples:
            self.known_names = names
            self.recognizer.train(samples, np.array(labels))
            self.trained = True
            print(f"[INFO] {len(names)} visage(s) chargé(s) avec {len(samples)} échantillons: {names}")
        else:
            print("[INFO] Aucun visage connu — mode détection uniquement.")

    def recognize(self, frame_bgr):
        """Retourne liste de (top, right, bottom, left, name) pour chaque visage."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results = []
        for (x, y, w, h) in faces:
            name = "Inconnu"
            if self.trained:
                face_roi = self._preprocess(gray[y:y + h, x:x + w])
                label, confidence = self.recognizer.predict(face_roi)
                if confidence < self.threshold:
                    name = self.known_names[label]
            results.append((y, x + w, y + h, x, name))
        return results

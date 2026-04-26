import os
import cv2
import numpy as np

MODELS_DIR = "models"
YUNET = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
COSINE_THRESHOLD = 0.363  # Seuil SFace : même personne si score > seuil


class FaceRecognizer:
    """Détection YuNet + reconnaissance SFace (deep learning, sans dlib)."""

    def __init__(self, known_faces_dir="known_faces", threshold=COSINE_THRESHOLD):
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.known = []  # [(name, feature_vector), ...]

        self.detector = cv2.FaceDetectorYN.create(
            YUNET, "", (320, 320), 0.6, 0.3, 5000
        )
        self.sf = cv2.FaceRecognizerSF.create(SFACE, "")
        self.load_known_faces()

    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.empty((0, 15), dtype=np.float32)

    def _feature(self, frame, face_row):
        try:
            aligned = self.sf.alignCrop(frame, face_row)
            return self.sf.feature(aligned)
        except cv2.error:
            return None

    def load_known_faces(self):
        self.known = []
        if not os.path.isdir(self.known_faces_dir):
            return
        for filename in sorted(os.listdir(self.known_faces_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(self.known_faces_dir, filename))
            if img is None:
                continue
            faces = self.detect_faces(img)
            if len(faces) == 0:
                continue
            feat = self._feature(img, faces[0])
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            self.known.append((name, feat))
        names = [n for n, _ in self.known]
        print(f"[INFO] {len(self.known)} visage(s) chargé(s): {names}")

    def recognize(self, frame):
        """Retourne [(x, y, w, h, name), ...] pour chaque visage détecté."""
        faces = self.detect_faces(frame)
        results = []
        for face in faces:
            x, y, w, h = (int(v) for v in face[:4])
            name = "Inconnu"
            if self.known:
                feat = self._feature(frame, face)
                if feat is not None:
                    best_score, best_name = -1.0, "Inconnu"
                    for kname, kfeat in self.known:
                        score = self.sf.match(feat, kfeat, 0)  # 0 = cosine
                        if score > best_score:
                            best_score, best_name = score, kname
                    if best_score >= self.threshold:
                        name = best_name
            results.append((x, y, w, h, name))
        return results

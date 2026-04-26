import os
import cv2
import numpy as np

MODELS_DIR = "models"
YUNET = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
COSINE_THRESHOLD = 0.363


class FaceRecognizer:
    """Détection YuNet + reconnaissance SFace multi-profils."""

    def __init__(self, known_faces_dir="known_faces", threshold=COSINE_THRESHOLD):
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.known = []  # [(name, [feat1, feat2, ...]), ...]

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

    def _load_image_feat(self, path):
        img = cv2.imread(path)
        if img is None:
            return None
        faces = self.detect_faces(img)
        if len(faces) == 0:
            return None
        return self._feature(img, faces[0])

    def load_known_faces(self):
        self.known = []
        if not os.path.isdir(self.known_faces_dir):
            return

        for item in sorted(os.listdir(self.known_faces_dir)):
            item_path = os.path.join(self.known_faces_dir, item)

            if os.path.isdir(item_path):
                # Multi-profils : dossier par personne
                feats = []
                for img_file in sorted(os.listdir(item_path)):
                    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    feat = self._load_image_feat(os.path.join(item_path, img_file))
                    if feat is not None:
                        feats.append(feat)
                if feats:
                    name = item.replace("_", " ").title()
                    self.known.append((name, feats))

            elif item.lower().endswith((".jpg", ".jpeg", ".png")):
                # Fichier unique (rétrocompatibilité)
                feat = self._load_image_feat(item_path)
                if feat is not None:
                    name = os.path.splitext(item)[0].replace("_", " ").title()
                    self.known.append((name, [feat]))

        names = [n for n, _ in self.known]
        total = sum(len(f) for _, f in self.known)
        print(f"[INFO] {len(self.known)} personne(s), {total} profil(s): {names}")

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
                    for kname, kfeats in self.known:
                        for kfeat in kfeats:
                            score = self.sf.match(feat, kfeat, 0)
                            if score > best_score:
                                best_score, best_name = score, kname
                    if best_score >= self.threshold:
                        name = best_name
            results.append((x, y, w, h, name))
        return results

import os
import face_recognition
import numpy as np
from PIL import Image


class FaceRecognizer:
    """Loads known faces and identifies them in frames."""

    def __init__(self, known_faces_dir="known_faces", tolerance=0.55):
        self.tolerance = tolerance
        self.known_encodings = []
        self.known_names = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory):
        if not os.path.isdir(directory):
            return
        for filename in os.listdir(directory):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                self.known_names.append(name)
        print(f"[INFO] {len(self.known_names)} visage(s) chargé(s): {self.known_names}")

    def recognize(self, frame_rgb):
        """Return list of (top, right, bottom, left, name) for each face found."""
        locations = face_recognition.face_locations(frame_rgb)
        encodings = face_recognition.face_encodings(frame_rgb, locations)
        results = []
        for encoding, location in zip(encodings, locations):
            name = "Inconnu"
            if self.known_encodings:
                matches = face_recognition.compare_faces(
                    self.known_encodings, encoding, tolerance=self.tolerance
                )
                distances = face_recognition.face_distance(self.known_encodings, encoding)
                if True in matches:
                    best = int(np.argmin(distances))
                    name = self.known_names[best]
            results.append((*location, name))
        return results

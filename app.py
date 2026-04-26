import os
import time
import threading
import cv2
from flask import Flask, render_template, Response, request, jsonify
from face_recognizer import FaceRecognizer

app = Flask(__name__)
os.makedirs("known_faces", exist_ok=True)

_lock = threading.Lock()
_frame = None
_results = []
_recognizer = FaceRecognizer()


def _capture_loop():
    global _frame, _results
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Caméra inaccessible")
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with _lock:
            _frame = frame.copy()
        idx += 1
        if idx % 4 == 0:
            results = _recognizer.recognize(frame)
            with _lock:
                _results = results


threading.Thread(target=_capture_loop, daemon=True).start()


def _annotate(frame, results):
    for (x, y, w, h, name) in results:
        color = (30, 200, 30) if name != "Inconnu" else (30, 30, 210)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ly = max(y - 8, th + 10)
        cv2.rectangle(frame, (x, ly - th - 8), (x + tw + 10, ly + 2), color, cv2.FILLED)
        cv2.putText(frame, name, (x + 5, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def _generate():
    while True:
        with _lock:
            frame = _frame.copy() if _frame is not None else None
            results = list(_results)
        if frame is None:
            time.sleep(0.033)
            continue
        _annotate(frame, results)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(_generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_face', methods=['POST'])
def add_face():
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Nom complet requis'}), 400

    with _lock:
        frame = _frame.copy() if _frame is not None else None
    if frame is None:
        return jsonify({'error': 'Caméra non disponible'}), 500

    faces = _recognizer.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({'error': 'Aucun visage détecté — rapprochez-vous de la caméra'}), 400

    filename = name.lower().replace(' ', '_') + '.jpg'
    cv2.imwrite(os.path.join('known_faces', filename), frame)
    _recognizer.load_known_faces()

    return jsonify({'success': True, 'name': name})


@app.route('/faces')
def get_faces():
    result = []
    for f in os.listdir('known_faces'):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            result.append(os.path.splitext(f)[0].replace('_', ' ').title())
    return jsonify(sorted(result))


@app.route('/threshold', methods=['GET', 'POST'])
def threshold():
    if request.method == 'POST':
        try:
            value = float(request.json.get('value', 0.363))
            value = max(0.1, min(0.9, value))
            _recognizer.threshold = value
            return jsonify({'success': True, 'threshold': value})
        except (TypeError, ValueError):
            return jsonify({'error': 'Valeur invalide'}), 400
    return jsonify({'threshold': _recognizer.threshold})


@app.route('/delete_face/<name>', methods=['DELETE'])
def delete_face(name):
    filename = name.lower().replace(' ', '_') + '.jpg'
    path = os.path.join('known_faces', filename)
    if os.path.exists(path):
        os.remove(path)
        _recognizer.load_known_faces()
        return jsonify({'success': True})
    return jsonify({'error': 'Introuvable'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

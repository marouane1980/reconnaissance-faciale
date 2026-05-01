"""
Analyse démographique des visages inconnus via DeepFace.
Tourne dans un thread séparé pour ne pas bloquer le flux vidéo.
"""

import cv2
import base64
import threading
import queue

try:
    from deepface import DeepFace
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_queue        = queue.Queue(maxsize=1)
_last_result  = []
_results_lock = threading.Lock()
_running      = False
_on_result_cb = None   # callback(analyses) appelé dès qu'un résultat est prêt

EMOTION_FR = {
    'happy': 'Heureux(se)', 'sad': 'Triste', 'angry': 'En colère',
    'fear': 'Peur', 'surprise': 'Surpris(e)', 'disgust': 'Dégoût',
    'neutral': 'Neutre', 'contempt': 'Mépris'
}


def _encode(img):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok or buf is None:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()


def _worker():
    global _last_result
    while True:
        # ── récupère un frame ──
        try:
            frame, bboxes = _queue.get(timeout=2)
        except queue.Empty:
            continue

        # ── analyse chaque visage inconnu ──
        analyses = []
        try:
            for (x, y, w, h) in bboxes:
                pad = int(max(w, h) * 0.15)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                # encode thumbnail — ne doit pas faire crasher le worker
                try:
                    face_b64 = _encode(face)
                except Exception:
                    face_b64 = None

                entry = {'bbox': [int(x), int(y), int(w), int(h)], 'face': face_b64}

                try:
                    res = DeepFace.analyze(
                        face,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    r = res[0] if isinstance(res, list) else res

                    gender_raw = r.get('dominant_gender', '') or ''
                    gender_fr  = 'Homme' if 'man' in gender_raw.lower() else 'Femme'
                    gender_scores = r.get('gender', {})
                    gkey = 'Man' if 'man' in gender_raw.lower() else 'Woman'
                    gender_conf = round(float(gender_scores.get(gkey, 0)), 1) \
                        if isinstance(gender_scores, dict) else None

                    emotion_raw = r.get('dominant_emotion', 'neutral') or 'neutral'
                    emotion_scores = r.get('emotion', {})
                    emotion_conf = round(float(emotion_scores.get(emotion_raw, 0)), 1) \
                        if isinstance(emotion_scores, dict) else None

                    age = int(r.get('age', 0))
                    if age <= 12:   tranche = 'Enfant'
                    elif age <= 17: tranche = 'Adolescent(e)'
                    elif age <= 35: tranche = 'Jeune adulte'
                    elif age <= 60: tranche = 'Adulte'
                    else:           tranche = 'Senior'

                    entry.update({
                        'age': age,
                        'age_range': tranche,
                        'gender': gender_fr,
                        'gender_conf': gender_conf,
                        'emotion': EMOTION_FR.get(emotion_raw, emotion_raw.capitalize()),
                        'emotion_conf': emotion_conf,
                        'face_size': _estimate_relative_size(w, frame.shape[1]),
                    })
                except Exception as e:
                    entry['error'] = str(e)

                analyses.append(entry)

        except Exception as e:
            print('[analyzer] erreur worker:', e)
        finally:
            with _results_lock:
                if analyses:   # ne réinitialise pas si aucun visage traité
                    _last_result = analyses
            if _on_result_cb and analyses:
                try:
                    _on_result_cb(analyses)
                except Exception:
                    pass
            _queue.task_done()


def _estimate_relative_size(face_w, frame_w):
    ratio = face_w / frame_w
    if ratio > 0.30:   return 'Très proche / visage large'
    elif ratio > 0.18: return 'Proche'
    elif ratio > 0.10: return 'Distance moyenne'
    else:              return 'Éloigné'


def set_on_result_callback(fn):
    global _on_result_cb
    _on_result_cb = fn


def start():
    global _running
    if not _AVAILABLE or _running:
        return
    _running = True
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def submit(frame, faces_results):
    """Envoie un frame à analyser (drop silencieux si déjà occupé)."""
    if not _AVAILABLE or not _running:
        return
    unknowns = [(x, y, w, h) for (x, y, w, h, name) in faces_results if name == 'Inconnu']
    if not unknowns:
        return
    try:
        _queue.put_nowait((frame.copy(), unknowns))
    except queue.Full:
        pass


def get_results():
    with _results_lock:
        return list(_last_result)


def is_available():
    return _AVAILABLE

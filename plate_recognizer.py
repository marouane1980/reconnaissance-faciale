"""Reconnaissance de plaques d'immatriculation + analyse IA des véhicules."""

import cv2
import numpy as np
import re
import threading
import queue
import time
import base64
from datetime import datetime

# ── OCR (easyocr, lazy init) ─────────────────────────────────────────────────
try:
    import easyocr as _easyocr_mod
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    _easyocr_mod   = None

_reader      = None
_reader_lock = threading.Lock()

# ── YOLO (ultralytics) pour type de véhicule ─────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO_CLS
    _YOLO_AVAILABLE = True
    _VEHICLE_CLASSES = {2: 'Voiture', 3: 'Moto', 5: 'Bus', 7: 'Camion'}
except ImportError:
    _YOLO_AVAILABLE = False
    _YOLO_CLS       = None
    _VEHICLE_CLASSES = {}

_yolo      = None
_yolo_lock = threading.Lock()

# ── Queue & résultats ─────────────────────────────────────────────────────────
_queue        = queue.Queue(maxsize=1)
_last_result  = []
_per_camera   = {}    # {cam_label: [results]}
_results_lock = threading.Lock()
_running      = False
_on_result_cb = None

# Pattern VIN (17 caractères, exclut I, O, Q)
_VIN_RE = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')

# ── Plages de couleurs HSV ───────────────────────────────────────────────────
_COLORS = [
    ('Blanc',   np.array([0,   0,   185]), np.array([180, 35,  255])),
    ('Noir',    np.array([0,   0,   0]),   np.array([180, 255, 55 ])),
    ('Gris',    np.array([0,   0,   55]),  np.array([180, 35,  185])),
    ('Rouge',   np.array([0,   100, 80]),  np.array([10,  255, 255])),
    ('Rouge',   np.array([160, 100, 80]),  np.array([180, 255, 255])),
    ('Bleu',    np.array([96,  80,  70]),  np.array([135, 255, 255])),
    ('Vert',    np.array([36,  80,  70]),  np.array([85,  255, 255])),
    ('Jaune',   np.array([18,  100, 100]), np.array([35,  255, 255])),
    ('Orange',  np.array([10,  100, 100]), np.array([18,  255, 255])),
    ('Marron',  np.array([8,   60,  40]),  np.array([20,  200, 140])),
    ('Beige',   np.array([18,  30,  170]), np.array([35,  80,  255])),
    ('Violet',  np.array([125, 60,  60]),  np.array([160, 255, 255])),
]


# ── Initialisations lazy ──────────────────────────────────────────────────────

def _init_reader():
    global _reader
    if not _OCR_AVAILABLE:
        return None
    with _reader_lock:
        if _reader is None:
            try:
                print('[plate] Chargement du modèle OCR easyocr...')
                _reader = _easyocr_mod.Reader(['fr', 'en'], gpu=False, verbose=False)
                print('[plate] OCR prêt')
            except Exception as e:
                print('[plate] OCR init error:', e)
    return _reader


def _init_yolo():
    global _yolo
    if not _YOLO_AVAILABLE:
        return None
    with _yolo_lock:
        if _yolo is None:
            try:
                _yolo = _YOLO_CLS('yolov8n.pt')
                print('[plate] YOLO prêt pour la classification véhicule')
            except Exception as e:
                print('[plate] YOLO init error:', e)
    return _yolo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _clean_plate(text):
    """Normalise le texte OCR → format plaque standardisé."""
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) < 4:
        return None
    # SIV français : AA-NNN-AA
    m = re.match(r'^([A-Z]{2})(\d{3})([A-Z]{2})$', text)
    if m:
        return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    # FNI : NNN-AA-NN
    m2 = re.match(r'^(\d{3,4})([A-Z]{1,3})(\d{2,3})$', text)
    if m2:
        return f'{m2.group(1)}-{m2.group(2)}-{m2.group(3)}'
    # Autre pays : retour brut si ≥ 5 chars
    return text if len(text) >= 5 else None


def _find_plate_regions(frame):
    """Détecte les régions candidates pour des plaques via morphologie."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 25, 180)

    # Fermeture horizontale pour relier les caractères d'une plaque
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        ratio = cw / ch
        area  = cw * ch
        if 2.0 < ratio < 9.0 and 1200 < area < (w * h * 0.12) and cw > 60:
            candidates.append((x, y, cw, ch))
    return candidates


def _detect_color(frame):
    """Couleur dominante du véhicule (zone haute = carrosserie)."""
    try:
        h, w = frame.shape[:2]
        roi  = frame[:max(1, int(h * 0.55)), :]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        best_name, best_pct = 'Inconnue', 0.0
        seen = set()
        for item in _COLORS:
            name = item[0]
            lo, hi = item[1], item[2]
            mask = cv2.inRange(hsv, lo, hi)
            pct  = mask.sum() / (roi.shape[0] * roi.shape[1] * 255) * 100
            if pct > best_pct:
                best_pct  = pct
                best_name = name
        return best_name, round(best_pct, 1)
    except Exception:
        return 'Inconnue', 0.0


def _detect_vehicle_type(frame):
    """Type de véhicule via YOLOv8 (voiture / camion / moto / bus)."""
    yolo = _init_yolo()
    if yolo is None:
        return None, 0
    try:
        results = yolo(frame, verbose=False, classes=list(_VEHICLE_CLASSES.keys()))[0]
        if len(results.boxes) == 0:
            return None, 0
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)
        idx   = int(confs.argmax())
        vtype = _VEHICLE_CLASSES.get(clss[idx], 'Véhicule')
        return vtype, round(float(confs[idx]) * 100)
    except Exception as e:
        print('[plate] YOLO error:', e)
        return None, 0


def _thumb(frame, width=180):
    try:
        h, w = frame.shape[:2]
        tw = width; th = int(h * tw / w)
        small = cv2.resize(frame, (tw, th))
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    except Exception:
        pass
    return None


# ── Worker thread ─────────────────────────────────────────────────────────────

def _worker():
    global _last_result
    reader = _init_reader()   # bloque jusqu'à chargement du modèle
    while True:
        try:
            frame, cam_label = _queue.get(timeout=2)
        except queue.Empty:
            continue

        results = []
        try:
            candidates = _find_plate_regions(frame)
            h, w = frame.shape[:2]

            for (x, y, cw, ch) in candidates:
                pad = 6
                x1  = max(0, x - pad);  y1 = max(0, y - pad)
                x2  = min(w, x + cw + pad); y2 = min(h, y + ch + pad)
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                # OCR
                plate_text = None
                conf       = 0
                if reader is not None:
                    try:
                        ocr_res = reader.readtext(
                            plate_crop,
                            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                            detail=1,
                        )
                        if ocr_res:
                            best      = max(ocr_res, key=lambda r: r[2])
                            plate_text = _clean_plate(best[1])
                            conf       = round(best[2] * 100)
                    except Exception as e:
                        print('[plate] OCR error:', e)

                if not plate_text:
                    continue

                # Miniature plaque
                ok, buf    = cv2.imencode('.jpg', plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                plate_b64  = ('data:image/jpeg;base64,' + base64.b64encode(buf).decode()) if ok else None

                # Analyse véhicule
                color, color_pct   = _detect_color(frame)
                vtype, vtype_conf  = _detect_vehicle_type(frame)

                results.append({
                    'plate':      plate_text,
                    'confidence': conf,
                    'bbox':       [int(x), int(y), int(cw), int(ch)],
                    'plate_img':  plate_b64,
                    'frame_img':  _thumb(frame),
                    'color':      color,
                    'color_pct':  color_pct,
                    'vtype':      vtype,
                    'vtype_conf': vtype_conf,
                    'camera':     cam_label,
                    'timestamp':  _now_str(),
                    'ts':         time.time(),
                })

        except Exception as e:
            print('[plate] worker error:', e)
        finally:
            with _results_lock:
                if results:
                    _last_result = results
                    if cam_label:
                        _per_camera[cam_label] = results
                elif cam_label:
                    _per_camera[cam_label] = []
            if _on_result_cb and results:
                for r in results:
                    try:
                        _on_result_cb(r)
                    except Exception:
                        pass
            _queue.task_done()


# ── API publique ──────────────────────────────────────────────────────────────

def start():
    global _running
    if _running:
        return
    if not _OCR_AVAILABLE:
        print('[plate] easyocr non disponible — reconnaissance plaques désactivée')
        return
    _running = True
    # Init OCR en tâche de fond (peut prendre 20–40 s)
    threading.Thread(target=_init_reader, daemon=True).start()
    threading.Thread(target=_worker,      daemon=True).start()
    print('[plate] module démarré')


def submit(frame, cam_label=''):
    if not _running:
        return
    try:
        _queue.put_nowait((frame.copy(), cam_label))
    except queue.Full:
        pass


def get_results():
    with _results_lock:
        return list(_last_result)


def get_results_for_camera(cam_label):
    with _results_lock:
        return list(_per_camera.get(cam_label, []))


def extract_vehicle_info(frame):
    """Analyse une image : OCR complet → plaque + n° chassis (VIN) + couleur + type."""
    reader = _init_reader()
    info = {
        'plate':     '',
        'chassis':   '',
        'color':     '',
        'vtype':     '',
        'plate_img': None,
        'raw_text':  [],
    }
    if frame is None:
        return info

    color, _ = _detect_color(frame)
    if color:
        info['color'] = color
    vtype, _ = _detect_vehicle_type(frame)
    if vtype:
        info['vtype'] = vtype

    if reader is None:
        return info

    try:
        ocr_res = reader.readtext(
            frame,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            detail=1,
        )
    except Exception as e:
        print('[plate] extract OCR error:', e)
        return info

    raw_lines = []
    plate_candidates = []
    for box, text, conf in ocr_res:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        raw_lines.append({'text': text, 'conf': round(float(conf) * 100)})

        # VIN détecté ?
        if not info['chassis']:
            m = _VIN_RE.search(clean)
            if m:
                info['chassis'] = m.group(1)

        # Plaque candidate ?
        plate_text = _clean_plate(text)
        if plate_text:
            plate_candidates.append((plate_text, float(conf), box))

    if plate_candidates:
        plate_candidates.sort(key=lambda r: r[1], reverse=True)
        info['plate'] = plate_candidates[0][0]
        # Recadre la zone de plaque pour la miniature
        try:
            box = plate_candidates[0][2]
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1, x2 = max(0, min(xs)), min(frame.shape[1], max(xs))
            y1, y2 = max(0, min(ys)), min(frame.shape[0], max(ys))
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    info['plate_img'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
        except Exception:
            pass

    info['raw_text'] = raw_lines
    return info


def set_on_result_callback(fn):
    global _on_result_cb
    _on_result_cb = fn


def is_available():
    return _OCR_AVAILABLE

"""Gestionnaire multi-caméras avec configuration persistante."""

import cv2
import json
import os
import time
import uuid
import threading

from face_recognizer import FaceRecognizer
import analyzer
import tracker
import behavior

CAMERAS_FILE = 'cameras.json'

ROOMS = [
    'Entrée', 'Salon', 'Salle à manger', 'Cuisine', 'Couloir',
    'Chambre principale', 'Chambre', 'Bureau', 'Salle de bain',
    'Garage', 'Jardin', 'Terrasse', 'Cave', 'Grenier', 'Parking', 'Autre'
]

FEATURES = {
    'face_recognition':  'Reconnaissance faciale',
    'behavior_analysis': 'Analyse comportementale',
}


def _default_cameras():
    return [{
        'id':       'cam_0',
        'label':    'Caméra 1',
        'rooms':    ['Entrée'],
        'url':      '0',
        'ip':       '',
        'port':     '',
        'protocol': 'webcam',
        'username': '',
        'password': '',
        'enabled':  True,
        'features': ['face_recognition', 'behavior_analysis'],
    }]


def _load_raw():
    if not os.path.exists(CAMERAS_FILE):
        cams = _default_cameras()
        with open(CAMERAS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'cameras': cams}, f, indent=2, ensure_ascii=False)
        return cams
    with open(CAMERAS_FILE, encoding='utf-8') as f:
        cams = json.load(f).get('cameras', [])
    # Backward compat: convert old 'room' (str) to 'rooms' (list)
    for cam in cams:
        if 'room' in cam and 'rooms' not in cam:
            cam['rooms'] = [cam.pop('room')] if cam['room'] else []
        cam.setdefault('rooms', [])
    return cams


def _save_raw(cameras):
    with open(CAMERAS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'cameras': cameras}, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════
#  Worker par caméra
# ════════════════════════════════════════════

class CameraWorker:
    def __init__(self, config):
        self.cam_id    = config['id']
        self.config    = dict(config)
        self.frame     = None
        self.results   = []
        self.connected = False
        self.error     = ''
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None
        self._rec      = FaceRecognizer()

    # ── url ──────────────────────────────────
    def _url(self):
        raw = str(self.config.get('url', '0')).strip()
        try:
            return int(raw)          # webcam local
        except ValueError:
            pass
        usr = self.config.get('username', '').strip()
        pwd = self.config.get('password', '').strip()
        if usr and pwd and raw.startswith('rtsp://') and '@' not in raw:
            raw = 'rtsp://' + usr + ':' + pwd + '@' + raw[7:]
        return raw

    # ── cycle de vie ─────────────────────────
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def reload_faces(self):
        self._rec.load_known_faces()

    # ── accesseurs thread-safe ────────────────
    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def get_results(self):
        with self._lock:
            return list(self.results)

    def detect_faces(self, frame):
        return self._rec.detect_faces(frame)

    # ── boucle principale ─────────────────────
    def _loop(self):
        url = self._url()
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            self.error     = "Impossible d'ouvrir : " + str(url)
            self.connected = False
            self._running  = False
            print(f'[cam {self.cam_id}]', self.error)
            return

        self.connected = True
        self.error     = ''
        print(f'[cam {self.cam_id}] connectée :', url)
        idx = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self._lock:
                self.frame = frame.copy()

            idx += 1
            if idx % 4 != 0:
                continue

            feats = set(self.config.get('features', []))
            label = self.config.get('label', self.cam_id)

            if 'face_recognition' in feats:
                res = self._rec.recognize(frame)
                with self._lock:
                    self.results = res
                analyzer.submit(frame, res)
                tracker.update(res, frame, analyzer.get_results(), cam_label=label)
                if 'behavior_analysis' in feats:
                    behavior.submit(frame, res)
            else:
                with self._lock:
                    self.results = []
                if 'behavior_analysis' in feats:
                    behavior.submit(frame)

        cap.release()
        self.connected = False
        print(f'[cam {self.cam_id}] arrêtée')

    # ── sérialisation ─────────────────────────
    def to_dict(self, safe=True):
        d = dict(self.config)
        d['connected'] = self.connected
        d['running']   = self._running
        d['error']     = self.error
        if safe:
            d.pop('password', None)
        return d


# ════════════════════════════════════════════
#  Manager principal
# ════════════════════════════════════════════

class CameraManager:
    def __init__(self):
        self._workers = {}       # {cam_id: CameraWorker}
        self._lock    = threading.Lock()
        self._boot()

    def _boot(self):
        for cfg in _load_raw():
            w = CameraWorker(cfg)
            self._workers[cfg['id']] = w
            if cfg.get('enabled', True):
                w.start()

    def _persist(self):
        _save_raw([w.config for w in self._workers.values()])

    # ── CRUD ──────────────────────────────────
    def list(self):
        with self._lock:
            return [w.to_dict() for w in self._workers.values()]

    def get(self, cam_id, safe=True):
        w = self._workers.get(cam_id)
        return w.to_dict(safe=safe) if w else None

    def add(self, data):
        cam_id = 'cam_' + uuid.uuid4().hex[:8]
        data['id'] = cam_id
        data.setdefault('enabled', True)
        data.setdefault('features', ['face_recognition'])
        with self._lock:
            w = CameraWorker(data)
            self._workers[cam_id] = w
            if data.get('enabled', True):
                w.start()
            self._persist()
        return cam_id

    def update(self, cam_id, data):
        with self._lock:
            w = self._workers.get(cam_id)
            if not w:
                return False
            w.stop()
            time.sleep(0.25)
            data['id'] = cam_id
            w.config   = data
            w.error    = ''
            if data.get('enabled', True):
                w.start()
            self._persist()
        return True

    def delete(self, cam_id):
        with self._lock:
            w = self._workers.pop(cam_id, None)
            if not w:
                return False
            w.stop()
            self._persist()
        return True

    def toggle(self, cam_id, enabled):
        with self._lock:
            w = self._workers.get(cam_id)
            if not w:
                return False
            w.config['enabled'] = enabled
            if enabled and not w._running:
                w.start()
            elif not enabled:
                w.stop()
            self._persist()
        return True

    # ── Helpers globaux ───────────────────────
    def reload_all_faces(self):
        with self._lock:
            for w in self._workers.values():
                w.reload_faces()

    def set_threshold(self, value):
        with self._lock:
            for w in self._workers.values():
                w._rec.threshold = value

    def get_threshold(self):
        w = next(iter(self._workers.values()), None)
        return w._rec.threshold if w else 0.363

    # ── Accès frame/résultats ─────────────────
    def get_frame(self, cam_id):
        w = self._workers.get(cam_id)
        return w.get_frame() if w else None

    def get_results(self, cam_id):
        w = self._workers.get(cam_id)
        return w.get_results() if w else []

    def detect_faces(self, cam_id, frame):
        w = self._workers.get(cam_id)
        return w.detect_faces(frame) if w else []

    def first_id(self):
        return next(iter(self._workers), None)

    def all_ids(self):
        return list(self._workers.keys())

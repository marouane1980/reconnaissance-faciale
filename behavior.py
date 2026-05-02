"""Analyse comportementale — détection de posture via MediaPipe Pose."""

import cv2
import base64
import time
import threading
import queue
from datetime import datetime

try:
    import mediapipe as mp
    _AVAILABLE    = True
    _mp_pose      = mp.solutions.pose
    _mp_draw      = mp.solutions.drawing_utils
    _mp_draw_sty  = mp.solutions.drawing_styles
    _pose_obj     = _mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _CONNECTIONS  = _mp_pose.POSE_CONNECTIONS
except Exception:
    _AVAILABLE   = False
    _pose_obj    = None
    _mp_draw     = None
    _mp_draw_sty = None
    _CONNECTIONS = None

_queue        = queue.Queue(maxsize=1)
_last_result  = []
_results_lock = threading.Lock()
_running      = False

# ── Paramètres ──
_show_landmarks = False
_tracked_poses  = {'debout', 'assis', 'allonge', 'course', 'saut', 'grimpe'}
_fall_detect    = True
_last_landmarks = None          # pour draw_landmarks_on()
_settings_lock  = threading.Lock()

# ── Historique de session ──
_history      = []
_history_lock = threading.Lock()
_current_sess = None
_hist_id      = 0
MAX_HISTORY   = 500

# ── Détection de chutes ──
_fall_history = []
_fall_lock    = threading.Lock()
_fall_id      = 0
_pose_window  = []              # [(ts, pose_key), ...]
FALL_WINDOW   = 2.5             # secondes

# Métadonnées postures : (label FR, couleur Tailwind, texte OpenCV)
_POSE_META = {
    'debout':  ('Debout',      'green',  'Debout'),
    'assis':   ('Assis(e)',    'blue',   'Assis'),
    'allonge': ('Allonge(e)', 'purple', 'Allonge'),
    'course':  ('En course',   'orange', 'Course'),
    'saut':    ('En saut',     'yellow', 'Saut'),
    'grimpe':  ('Grimpe',      'red',    'Grimpe'),
    'inconnu': ('Posture ?',   'slate',  '?'),
}


# ════════════════════════════════════════════
#  Helpers internes
# ════════════════════════════════════════════

def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _thumb(frame):
    """Miniature 120 px de large encodée en base64 JPEG."""
    try:
        h, w = frame.shape[:2]
        tw, th = 120, int(h * 120 / w)
        small = cv2.resize(frame, (tw, th))
        ok, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 65])
        if ok and buf is not None:
            return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    except Exception:
        pass
    return None


def _classify(lm):
    """Heuristique de classification. y=0 en haut, y=1 en bas (MediaPipe)."""
    nose         = lm[0]
    l_sh, r_sh   = lm[11], lm[12]
    l_hip, r_hip = lm[23], lm[24]
    l_kn, r_kn   = lm[25], lm[26]
    l_an, r_an   = lm[27], lm[28]
    l_wr, r_wr   = lm[15], lm[16]

    if any(p.visibility < 0.4 for p in [l_sh, r_sh, l_hip, r_hip]):
        return 'inconnu', 0.50

    sh_y  = (l_sh.y  + r_sh.y)  / 2
    hip_y = (l_hip.y + r_hip.y) / 2
    kn_y  = (l_kn.y  + r_kn.y)  / 2
    an_y  = (l_an.y  + r_an.y)  / 2
    wr_y  = (l_wr.y  + r_wr.y)  / 2

    body_span = an_y - sh_y

    if body_span < 0.22:               return 'allonge', 0.88
    if an_y <= hip_y + 0.08:          return 'saut',    0.80
    if wr_y < nose.y - 0.05:         return 'grimpe',  0.78
    if (kn_y - hip_y) < 0.10:        return 'assis',   0.82
    if abs(l_kn.y - r_kn.y) > 0.11:  return 'course',  0.72
    return 'debout', 0.90


def _check_fall(pose_key, ts, frame=None):
    """Détecte une chute : transition debout/course → allongé en < FALL_WINDOW s."""
    global _pose_window, _fall_id
    _pose_window = [(t, p) for t, p in _pose_window if ts - t <= FALL_WINDOW]
    _pose_window.append((ts, pose_key))

    if not _fall_detect or pose_key != 'allonge' or len(_pose_window) < 2:
        return
    last_upright = max(
        (t for t, p in _pose_window if p in ('debout', 'course')),
        default=None
    )
    if last_upright and ts - last_upright < FALL_WINDOW:
        name = _current_sess['_name'] if _current_sess else 'Inconnu'
        with _fall_lock:
            _fall_id += 1
            _fall_history.insert(0, {
                'id':        _fall_id,
                'name':      name,
                'timestamp': _now_str(),
                'photo':     _thumb(frame) if frame is not None else None,
            })
            if len(_fall_history) > 100:
                _fall_history.pop()
        print('[behavior] CHUTE DÉTECTÉE —', name, _now_str())


def _finalize(end_ts):
    global _current_sess, _hist_id
    if _current_sess is None:
        return
    duration = int(end_ts - _current_sess['_ts'])
    if duration >= 2 and _current_sess['behavior'] != 'inconnu':
        with _history_lock:
            _hist_id += 1
            _history.insert(0, {
                'id':         _hist_id,
                'name':       _current_sess['_name'],
                'photo':      _current_sess.get('_photo'),
                'behavior':   _current_sess['behavior'],
                'label':      _current_sess['label'],
                'color':      _current_sess['color'],
                'first_seen': _current_sess['first_seen'],
                'last_seen':  _now_str(),
                'duration_s': duration,
            })
            if len(_history) > MAX_HISTORY:
                _history.pop()
    _current_sess = None


def _update_session(pose_key, name, frame=None):
    global _current_sess
    ts  = time.time()
    now = _now_str()
    label, color, _ = _POSE_META.get(pose_key, _POSE_META['inconnu'])
    if _current_sess is None:
        _current_sess = {
            '_ts': ts, '_name': name,
            '_photo': _thumb(frame) if frame is not None else None,
            'behavior': pose_key, 'label': label, 'color': color,
            'first_seen': now,
        }
    elif _current_sess['behavior'] != pose_key:
        _finalize(ts)
        _current_sess = {
            '_ts': ts, '_name': name,
            '_photo': _thumb(frame) if frame is not None else None,
            'behavior': pose_key, 'label': label, 'color': color,
            'first_seen': now,
        }
    else:
        _current_sess['_name'] = name


# ════════════════════════════════════════════
#  Worker thread
# ════════════════════════════════════════════

def _worker():
    global _last_result, _last_landmarks
    while True:
        try:
            frame, name = _queue.get(timeout=2)
        except queue.Empty:
            continue

        results = []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = _pose_obj.process(rgb)
            if out.pose_landmarks:
                _last_landmarks = out.pose_landmarks
                lm = out.pose_landmarks.landmark
                pose_key, conf = _classify(lm)

                # Filtre par postures suivies
                with _settings_lock:
                    tracked = set(_tracked_poses)
                if pose_key not in tracked:
                    pose_key = 'inconnu'
                    conf     = 0.5

                label, color, vid_text = _POSE_META.get(pose_key, _POSE_META['inconnu'])
                ts = time.time()

                results.append({
                    'pose':       pose_key,
                    'label':      label,
                    'color':      color,
                    'vid_text':   vid_text,
                    'confidence': round(conf * 100),
                    'name':       name,
                })
                _update_session(pose_key, name, frame)
                _check_fall(pose_key, ts, frame)
            else:
                _last_landmarks = None
        except Exception as e:
            print('[behavior] worker error:', e)
        finally:
            with _results_lock:
                if results:
                    _last_result = results
            _queue.task_done()


# ════════════════════════════════════════════
#  API publique
# ════════════════════════════════════════════

def start():
    global _running
    if not _AVAILABLE or _running:
        return
    _running = True
    threading.Thread(target=_worker, daemon=True).start()


def submit(frame, face_results=None):
    if not _AVAILABLE or not _running:
        return
    name = 'Inconnu'
    if face_results:
        known = [n for (_, _, _, _, n) in face_results if n != 'Inconnu']
        if len(known) == 1:
            name = known[0]
        elif len(face_results) == 1:
            name = face_results[0][4]
    try:
        _queue.put_nowait((frame.copy(), name))
    except queue.Full:
        pass


def draw_landmarks_on(frame):
    """Dessine les landmarks sur la frame en-place si activé."""
    if not _show_landmarks or _last_landmarks is None or not _AVAILABLE:
        return
    try:
        _mp_draw.draw_landmarks(
            frame,
            _last_landmarks,
            _CONNECTIONS,
            landmark_drawing_spec=_mp_draw_sty.get_default_pose_landmarks_style(),
        )
    except Exception:
        pass


def get_results():
    with _results_lock:
        return list(_last_result)


def get_history(limit=500):
    with _history_lock:
        return list(_history[:limit])


def get_fall_history(limit=100):
    with _fall_lock:
        return list(_fall_history[:limit])


def clear_history():
    global _history, _hist_id, _current_sess
    with _history_lock:
        _history = []
        _hist_id = 0
    _current_sess = None


def clear_fall_history():
    global _fall_history, _fall_id
    with _fall_lock:
        _fall_history = []
        _fall_id = 0


def get_settings():
    with _settings_lock:
        return {
            'show_landmarks': _show_landmarks,
            'tracked_poses':  list(_tracked_poses),
            'fall_detect':    _fall_detect,
        }


def apply_settings(data):
    global _show_landmarks, _tracked_poses, _fall_detect
    with _settings_lock:
        if 'show_landmarks' in data:
            _show_landmarks = bool(data['show_landmarks'])
        if 'tracked_poses' in data:
            valid = {'debout', 'assis', 'allonge', 'course', 'saut', 'grimpe'}
            _tracked_poses = valid & set(data['tracked_poses'])
            if not _tracked_poses:        # au moins une posture doit rester active
                _tracked_poses = valid
        if 'fall_detect' in data:
            _fall_detect = bool(data['fall_detect'])


def is_available():
    return _AVAILABLE

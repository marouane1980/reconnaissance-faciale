"""Analyse comportementale — détection de posture via MediaPipe Pose.

Refonte :
  - Normalisation par la taille du tronc (invariant à la distance caméra).
  - Classification basée sur :
      * angle du tronc vs. verticale (épaules → hanches)
      * angle des genoux (hanche-genou-cheville)
      * positions relatives, pas absolues
  - Hystérésis temporelle (vote majoritaire sur N frames) pour éviter le flickering.
  - Détection de chute multi-critères :
      * Transition rapide tronc vertical → tronc horizontal (< 1.5 s)
      * Vélocité verticale du centre de gravité au-dessus d'un seuil
      * Immobilité confirmée pendant ~1.5 s après la chute
      * Cooldown 30 s entre deux alertes
  - draw_landmarks_on() conservé.
"""

import cv2
import math
import base64
import time
import threading
import queue
import logging
from collections import deque
from datetime import datetime

log = logging.getLogger('faceid.behavior')

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
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
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
_tracked_poses  = {'debout', 'assis', 'allonge', 'course', 'saut', 'grimpe', 'penche'}
_fall_detect    = True
_last_landmarks = None
_settings_lock  = threading.Lock()

# ── Historique ──
_history      = []
_history_lock = threading.Lock()
_current_sess = None
_hist_id      = 0
MAX_HISTORY   = 500

# ── Détection de chute ──
_fall_history       = []
_fall_lock          = threading.Lock()
_fall_id            = 0
_last_fall_alert_ts = 0.0
FALL_COOLDOWN       = 30.0     # s entre deux alertes
FALL_DROP_WINDOW    = 1.5      # s pour passer vertical → horizontal
FALL_STILL_WINDOW   = 1.5      # s d'immobilité confirmée après
FALL_MIN_VELOCITY   = 0.35     # vitesse verticale min (frac/s) — y_norm dérivé du tronc
FALL_STILL_VAR      = 0.0030   # variance des positions du COG pour considérer immobile

# ── Buffers temporels ──
_VOTE_FRAMES   = 7             # vote majoritaire sur 7 frames (~0,5–1 s @ 8–14 fps)
_RAW_BUFFER    = deque(maxlen=_VOTE_FRAMES)         # [(pose, conf)]
_TRACK_BUFFER  = deque(maxlen=60)                   # [(ts, cog_y_norm, tilt_deg)]

_POSE_META = {
    'debout':  ('Debout',     'green',  'Debout'),
    'assis':   ('Assis(e)',   'blue',   'Assis'),
    'allonge': ('Allonge(e)', 'purple', 'Allonge'),
    'penche':  ('Penche(e)',  'cyan',   'Penche'),
    'course':  ('En course',  'orange', 'Course'),
    'saut':    ('En saut',    'yellow', 'Saut'),
    'grimpe':  ('Grimpe',     'red',    'Grimpe'),
    'inconnu': ('Posture ?',  'slate',  '?'),
}


# ════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════

def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _thumb(frame):
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


def _vis(p, thr=0.4):
    return getattr(p, 'visibility', 1.0) >= thr


def _midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _angle3(a, b, c):
    """Angle ABC en degrés (B sommet)."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    d1 = math.hypot(bax, bay) or 1e-6
    d2 = math.hypot(bcx, bcy) or 1e-6
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (d1 * d2)))
    return math.degrees(math.acos(cosv))


def _torso_tilt_deg(shoulders_mid, hips_mid):
    """Inclinaison du tronc par rapport à la verticale (0° = debout, 90° = couché)."""
    dx = hips_mid[0] - shoulders_mid[0]
    dy = hips_mid[1] - shoulders_mid[1]
    # axe vertical = (0, 1) dans le repère image
    return abs(math.degrees(math.atan2(dx, max(dy, 1e-6))))


# ════════════════════════════════════════════
#  Classification
# ════════════════════════════════════════════

def _classify(lm):
    """Retourne (pose_key, confidence, metrics_dict).

    Robust aux distances et angles. Combine :
      - tilt du tronc (degrés vs. verticale)
      - angle moyen des genoux (hanche-genou-cheville)
      - positions relatives bras/jambes par rapport au tronc
    """
    nose         = lm[0]
    l_sh, r_sh   = lm[11], lm[12]
    l_hip, r_hip = lm[23], lm[24]
    l_kn, r_kn   = lm[25], lm[26]
    l_an, r_an   = lm[27], lm[28]
    l_wr, r_wr   = lm[15], lm[16]

    # Visibilité minimale du tronc
    if not (_vis(l_sh) and _vis(r_sh) and _vis(l_hip) and _vis(r_hip)):
        return 'inconnu', 0.40, {}

    sh_mid  = _midpoint(l_sh, r_sh)
    hip_mid = _midpoint(l_hip, r_hip)

    torso_h = max(0.02, math.hypot(hip_mid[0] - sh_mid[0], hip_mid[1] - sh_mid[1]))
    tilt    = _torso_tilt_deg(sh_mid, hip_mid)

    cog_y = (sh_mid[1] + hip_mid[1]) / 2.0    # centre de gravité approx (en y normalisé)

    metrics = {
        'tilt_deg':  round(tilt, 1),
        'torso_h':   round(torso_h, 3),
        'cog_y':     round(cog_y, 3),
    }

    # ── 1) Allongé : tronc presque horizontal ──
    if tilt > 55:
        return 'allonge', 0.90, metrics

    # ── 2) Penché : tronc significativement incliné mais pas horizontal ──
    if tilt > 35:
        # On distingue penché de assis grâce à l'angle des genoux (genoux pliés ⇒ assis-penché vers l'avant)
        knees_visible = _vis(l_kn) and _vis(r_kn) and _vis(l_an) and _vis(r_an)
        if knees_visible:
            la = _angle3((l_hip.x, l_hip.y), (l_kn.x, l_kn.y), (l_an.x, l_an.y))
            ra = _angle3((r_hip.x, r_hip.y), (r_kn.x, r_kn.y), (r_an.x, r_an.y))
            knee_angle = (la + ra) / 2.0
            metrics['knee_angle'] = round(knee_angle, 1)
            if knee_angle < 130:
                return 'assis', 0.78, metrics
        return 'penche', 0.72, metrics

    # ── Tronc à peu près vertical ──
    knees_visible = _vis(l_kn) and _vis(r_kn) and _vis(l_an) and _vis(r_an)
    if not knees_visible:
        # Sans genoux visibles, on suppose debout par défaut (cadrage haut-buste)
        # mais on vérifie quand même les bras pour grimpe.
        if _vis(l_wr) and _vis(r_wr) and (l_wr.y < nose.y - 0.05 and r_wr.y < nose.y - 0.05):
            return 'grimpe', 0.70, metrics
        return 'debout', 0.78, metrics

    la = _angle3((l_hip.x, l_hip.y), (l_kn.x, l_kn.y), (l_an.x, l_an.y))
    ra = _angle3((r_hip.x, r_hip.y), (r_kn.x, r_kn.y), (r_an.x, r_an.y))
    knee_angle = (la + ra) / 2.0
    knee_diff  = abs(la - ra)
    metrics['knee_angle'] = round(knee_angle, 1)
    metrics['knee_diff']  = round(knee_diff,  1)

    # ── 3) Saut : les deux chevilles au-dessus des genoux (en y, donc plus haut dans l'image) ──
    an_y_avg  = (l_an.y + r_an.y) / 2.0
    kn_y_avg  = (l_kn.y + r_kn.y) / 2.0
    if an_y_avg < kn_y_avg - 0.05:
        return 'saut', 0.78, metrics

    # ── 4) Grimpe : poignets clairement plus haut que le nez ──
    if _vis(l_wr) and _vis(r_wr) and l_wr.y < nose.y - 0.05 and r_wr.y < nose.y - 0.05:
        return 'grimpe', 0.78, metrics

    # ── 5) Assis : genoux pliés (angle ≪ 180°) ──
    #     - Référence : assis sur chaise ≈ 90–110°, jambes droites ≈ 165–180°.
    if knee_angle < 140:
        return 'assis', 0.85, metrics

    # ── 6) Course : grosse différence d'angles entre les deux jambes (asymétrie de marche) ──
    #     - Pour éviter les faux positifs « jambe levée », on demande aussi un petit penchement avant
    #       implicitement absent ici (tilt < 35), donc on relève le seuil.
    if knee_diff > 35 and knee_angle < 165:
        return 'course', 0.70, metrics

    # ── 7) Debout par défaut ──
    return 'debout', 0.90, metrics


def _vote_majority():
    """Vote majoritaire pondéré par confiance sur _RAW_BUFFER."""
    if not _RAW_BUFFER:
        return 'inconnu', 0.40
    scores = {}
    for pose, conf in _RAW_BUFFER:
        scores[pose] = scores.get(pose, 0.0) + conf
    best = max(scores.items(), key=lambda kv: kv[1])
    pose = best[0]
    # Confiance moyenne sur les frames de cette pose
    matches = [c for p, c in _RAW_BUFFER if p == pose]
    avg = sum(matches) / max(1, len(matches))
    # Pénalité si la pose dominante est en minorité
    ratio = len(matches) / len(_RAW_BUFFER)
    return pose, max(0.4, avg * (0.6 + 0.4 * ratio))


# ════════════════════════════════════════════
#  Détection de chute
# ════════════════════════════════════════════

def _check_fall(stable_pose, ts, cog_y, tilt, frame, name):
    """Une vraie chute = tronc vertical → horizontal en peu de temps + vitesse + immobilité."""
    global _fall_id, _last_fall_alert_ts

    if not _fall_detect:
        return
    if ts - _last_fall_alert_ts < FALL_COOLDOWN:
        return
    if stable_pose != 'allonge':
        return

    # Cherche un point récent où le tronc était nettement vertical
    upright_pt = None
    for (t, _cog, _tilt) in _TRACK_BUFFER:
        if ts - t > FALL_DROP_WINDOW:
            continue
        if _tilt < 25:                      # tronc bien vertical
            upright_pt = (t, _cog, _tilt)
            break
    if upright_pt is None:
        return

    t0, cog0, _ = upright_pt
    dt = max(ts - t0, 0.05)
    drop = (cog_y - cog0) / dt              # > 0 si descente (y augmente)
    if drop < FALL_MIN_VELOCITY:
        return

    # Confirmation d'immobilité dans les FALL_STILL_WINDOW secondes précédentes
    recent_cog = [c for (t, c, _t) in _TRACK_BUFFER if ts - t <= FALL_STILL_WINDOW]
    if len(recent_cog) >= 4:
        m = sum(recent_cog) / len(recent_cog)
        var = sum((c - m) ** 2 for c in recent_cog) / len(recent_cog)
        if var > FALL_STILL_VAR * 4:        # encore en mouvement → on attend un autre tick
            return

    # ✓ Toutes les conditions remplies
    _last_fall_alert_ts = ts
    with _fall_lock:
        _fall_id += 1
        _fall_history.insert(0, {
            'id':        _fall_id,
            'name':      name,
            'timestamp': _now_str(),
            'photo':     _thumb(frame) if frame is not None else None,
            'velocity':  round(drop, 3),
            'tilt_deg':  round(tilt, 1),
        })
        if len(_fall_history) > 100:
            _fall_history.pop()
    log.warning('CHUTE DÉTECTÉE name=%s velocity=%.3f tilt=%.1f', name, drop, tilt)


# ════════════════════════════════════════════
#  Session / historique
# ════════════════════════════════════════════

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
                'camera':     _current_sess.get('_cam', ''),
            })
            if len(_history) > MAX_HISTORY:
                _history.pop()
    _current_sess = None


def _update_session(pose_key, name, frame=None, cam_label=''):
    global _current_sess
    ts  = time.time()
    now = _now_str()
    label, color, _ = _POSE_META.get(pose_key, _POSE_META['inconnu'])
    if _current_sess is None:
        _current_sess = {
            '_ts': ts, '_name': name, '_cam': cam_label,
            '_photo': _thumb(frame) if frame is not None else None,
            'behavior': pose_key, 'label': label, 'color': color,
            'first_seen': now,
        }
    elif _current_sess['behavior'] != pose_key:
        _finalize(ts)
        _current_sess = {
            '_ts': ts, '_name': name, '_cam': cam_label,
            '_photo': _thumb(frame) if frame is not None else None,
            'behavior': pose_key, 'label': label, 'color': color,
            'first_seen': now,
        }
    else:
        _current_sess['_name'] = name
        _current_sess['_cam']  = cam_label


# ════════════════════════════════════════════
#  Worker thread
# ════════════════════════════════════════════

def _worker():
    global _last_result, _last_landmarks
    while True:
        try:
            frame, name, cam_label = _queue.get(timeout=2)
        except queue.Empty:
            continue

        results = []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = _pose_obj.process(rgb)
            if out.pose_landmarks:
                _last_landmarks = out.pose_landmarks
                lm = out.pose_landmarks.landmark
                pose_raw, conf_raw, metrics = _classify(lm)

                # Buffer pour vote majoritaire
                _RAW_BUFFER.append((pose_raw, conf_raw))
                pose_key, conf = _vote_majority()

                # Filtre par postures suivies
                with _settings_lock:
                    tracked = set(_tracked_poses)
                if pose_key not in tracked:
                    pose_key = 'inconnu'
                    conf     = 0.5

                # Suivi temporel pour la chute
                ts = time.time()
                if 'cog_y' in metrics and 'tilt_deg' in metrics:
                    _TRACK_BUFFER.append((ts, metrics['cog_y'], metrics['tilt_deg']))
                    _check_fall(pose_key, ts, metrics['cog_y'], metrics['tilt_deg'], frame, name)

                label, color, vid_text = _POSE_META.get(pose_key, _POSE_META['inconnu'])
                results.append({
                    'pose':       pose_key,
                    'label':      label,
                    'color':      color,
                    'vid_text':   vid_text,
                    'confidence': round(conf * 100),
                    'name':       name,
                    'metrics':    metrics,
                })
                _update_session(pose_key, name, frame, cam_label)
            else:
                _last_landmarks = None
                _RAW_BUFFER.clear()
        except Exception as e:
            log.exception('worker error: %s', e)
        finally:
            with _results_lock:
                if results:
                    _last_result = results
            _queue.task_done()


# ════════════════════════════════════════════
#  API publique (inchangée)
# ════════════════════════════════════════════

def start():
    global _running
    if not _AVAILABLE or _running:
        return
    _running = True
    threading.Thread(target=_worker, daemon=True).start()
    log.info('module behavior démarré (mediapipe pose)')


def submit(frame, face_results=None, cam_label=''):
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
        _queue.put_nowait((frame.copy(), name, cam_label))
    except queue.Full:
        pass


def draw_landmarks_on(frame):
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
    global _fall_history, _fall_id, _last_fall_alert_ts
    with _fall_lock:
        _fall_history = []
        _fall_id = 0
    _last_fall_alert_ts = 0.0


def get_settings():
    with _settings_lock:
        return {
            'show_landmarks': _show_landmarks,
            'tracked_poses':  list(_tracked_poses),
            'fall_detect':    _fall_detect,
            'vote_frames':    _VOTE_FRAMES,
        }


def apply_settings(data):
    global _show_landmarks, _tracked_poses, _fall_detect
    with _settings_lock:
        if 'show_landmarks' in data:
            _show_landmarks = bool(data['show_landmarks'])
        if 'tracked_poses' in data:
            valid = {'debout', 'assis', 'allonge', 'penche', 'course', 'saut', 'grimpe'}
            _tracked_poses = valid & set(data['tracked_poses'])
            if not _tracked_poses:
                _tracked_poses = valid
        if 'fall_detect' in data:
            _fall_detect = bool(data['fall_detect'])


def is_available():
    return _AVAILABLE

"""Classifieur ML supervisé pour postures, basé sur MediaPipe Pose.

Architecture :
  1. extract_features(landmarks) → vecteur de features instantanées (par frame)
  2. FeatureBuffer accumule les features sur une fenêtre glissante
  3. window_features(buffer) → vecteur final (mean/std/velocity/range)
  4. predict(features) utilise un modèle sklearn pickle si dispo, sinon None

Le modèle est entraîné offline par `train_behavior.py` à partir de clips JSONL
enregistrés via l'API /behavior/record.
"""

import os
import math
import pickle
import logging
from collections import deque

log = logging.getLogger('faceid.behavior_clf')

MODEL_PATH = os.path.join('models', 'behavior_classifier.pkl')

# Indices MediaPipe Pose
NOSE       = 0
L_SHO, R_SHO = 11, 12
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


# ════════════════════════════════════════════
#  Géométrie
# ════════════════════════════════════════════

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle3(a, b, c):
    """Angle ABC en degrés (B sommet)."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    d1 = math.hypot(bax, bay) or 1e-6
    d2 = math.hypot(bcx, bcy) or 1e-6
    cosv = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / (d1 * d2)))
    return math.degrees(math.acos(cosv))


def _torso_tilt_deg(sh_mid, hip_mid):
    dx = hip_mid[0] - sh_mid[0]
    dy = hip_mid[1] - sh_mid[1]
    return abs(math.degrees(math.atan2(dx, max(dy, 1e-6))))


# ════════════════════════════════════════════
#  Features par frame (28 dims)
# ════════════════════════════════════════════

FRAME_FEATURE_NAMES = [
    'tilt_deg',
    'torso_h',
    'knee_angle_l', 'knee_angle_r', 'knee_angle_mean', 'knee_angle_diff',
    'elbow_angle_l', 'elbow_angle_r',
    'cog_y', 'cog_x',
    'nose_y_rel', 'wrist_l_y_rel', 'wrist_r_y_rel',
    'hip_y_rel', 'knee_y_rel', 'ankle_y_rel',
    'hip_knee_dist', 'knee_ankle_dist',
    'shoulders_w', 'hips_w',
    'aspect_ratio',
    'leg_symmetry',
    'arms_above_head',
    'feet_above_knees',
    'vis_lower',
    'vis_upper',
    'span_y',
    'span_x',
]


def extract_features(lm, vis_threshold=0.4):
    """Extrait un vecteur de features depuis les landmarks MediaPipe.

    `lm` est la liste pose_landmarks.landmark de MediaPipe (33 points).
    Retourne (features: list[float], visible: bool).
    Si le tronc n'est pas visible, retourne ([], False).
    """
    def pt(i):
        p = lm[i]
        return (p.x, p.y), getattr(p, 'visibility', 1.0)

    sh_l, vsl = pt(L_SHO); sh_r, vsr = pt(R_SHO)
    hp_l, vhl = pt(L_HIP); hp_r, vhr = pt(R_HIP)
    if min(vsl, vsr, vhl, vhr) < vis_threshold:
        return [], False

    no, _   = pt(NOSE)
    el_l, vel = pt(L_ELB); el_r, ver = pt(R_ELB)
    wr_l, vwl = pt(L_WRI); wr_r, vwr = pt(R_WRI)
    kn_l, vkl = pt(L_KNE); kn_r, vkr = pt(R_KNE)
    an_l, val = pt(L_ANK); an_r, var = pt(R_ANK)

    sh_mid  = _midpoint(sh_l, sh_r)
    hip_mid = _midpoint(hp_l, hp_r)
    kn_mid  = _midpoint(kn_l, kn_r)
    an_mid  = _midpoint(an_l, an_r)
    wr_mid  = _midpoint(wr_l, wr_r)

    torso_h = max(0.02, _dist(sh_mid, hip_mid))
    tilt    = _torso_tilt_deg(sh_mid, hip_mid)

    knee_l = _angle3(hp_l, kn_l, an_l)
    knee_r = _angle3(hp_r, kn_r, an_r)
    elb_l  = _angle3(sh_l, el_l, wr_l) if min(vel, vwl) > vis_threshold else 180.0
    elb_r  = _angle3(sh_r, el_r, wr_r) if min(ver, vwr) > vis_threshold else 180.0

    cog = ((sh_mid[0] + hip_mid[0]) / 2.0, (sh_mid[1] + hip_mid[1]) / 2.0)

    # Toutes les positions y sont normalisées par la hauteur du tronc (relatif au cog)
    def rel_y(p): return (p[1] - cog[1]) / torso_h
    def rel_x(p): return (p[0] - cog[0]) / torso_h

    span_y = abs(an_mid[1] - sh_mid[1]) / torso_h
    span_x = abs(max(sh_l[0], sh_r[0], hp_l[0], hp_r[0], an_l[0], an_r[0]) -
                 min(sh_l[0], sh_r[0], hp_l[0], hp_r[0], an_l[0], an_r[0])) / torso_h

    feats = [
        tilt,
        torso_h,
        knee_l, knee_r, (knee_l + knee_r) / 2.0, abs(knee_l - knee_r),
        elb_l, elb_r,
        cog[1], cog[0],
        rel_y(no),
        rel_y(wr_l), rel_y(wr_r),
        rel_y(hip_mid),
        rel_y(kn_mid),
        rel_y(an_mid),
        _dist(hip_mid, kn_mid) / torso_h,
        _dist(kn_mid, an_mid) / torso_h,
        _dist(sh_l, sh_r) / torso_h,
        _dist(hp_l, hp_r) / torso_h,
        span_y / max(span_x, 0.01),                       # >1 = vertical, <1 = horizontal
        abs(knee_l - knee_r) / 180.0,                     # asymétrie normalisée
        1.0 if (wr_l[1] < no[1] and wr_r[1] < no[1]) else 0.0,
        1.0 if (an_mid[1] < kn_mid[1] - 0.05) else 0.0,   # pieds au-dessus des genoux
        (vkl + vkr + val + var) / 4.0,                    # visibilité bas du corps
        (vsl + vsr + vhl + vhr) / 4.0,                    # visibilité haut du corps
        span_y,
        span_x,
    ]
    assert len(feats) == len(FRAME_FEATURE_NAMES)
    return feats, True


# ════════════════════════════════════════════
#  Fenêtre temporelle
# ════════════════════════════════════════════

class FeatureBuffer:
    """Buffer circulaire de features par frame, agrège en vecteur final."""

    def __init__(self, window=15):
        self.window  = window
        self.frames  = deque(maxlen=window)   # [(ts, features)]

    def push(self, ts, features):
        self.frames.append((ts, features))

    def is_ready(self):
        return len(self.frames) >= max(5, self.window // 2)

    def aggregate(self):
        """mean, std, range, velocity → 4× FRAME_FEATURE_NAMES."""
        if not self.is_ready():
            return None
        n = len(self.frames)
        cols = len(FRAME_FEATURE_NAMES)
        # Transpose
        T = [[self.frames[i][1][c] for i in range(n)] for c in range(cols)]
        ts = [self.frames[i][0]    for i in range(n)]
        dt = max(ts[-1] - ts[0], 1e-3)

        out = []
        for col in T:
            m  = sum(col) / n
            sd = (sum((v - m) ** 2 for v in col) / n) ** 0.5
            rg = max(col) - min(col)
            vel = (col[-1] - col[0]) / dt
            out.extend([m, sd, rg, vel])
        return out

    def clear(self):
        self.frames.clear()


def aggregated_feature_names():
    out = []
    for n in FRAME_FEATURE_NAMES:
        out.extend([n + '_mean', n + '_std', n + '_range', n + '_vel'])
    return out


# ════════════════════════════════════════════
#  Modèle (chargement / prédiction)
# ════════════════════════════════════════════

_model       = None
_model_meta  = None        # {'classes': [...], 'feature_names': [...]}
_model_lock  = None

def load_model(path=MODEL_PATH):
    """Charge le modèle pickle s'il existe. Retourne True si chargé."""
    global _model, _model_meta
    if not os.path.exists(path):
        log.info('aucun modèle entraîné trouvé à %s — fallback heuristiques', path)
        return False
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        _model      = obj.get('model')
        _model_meta = {
            'classes':       list(getattr(_model, 'classes_', obj.get('classes', []))),
            'feature_names': obj.get('feature_names', aggregated_feature_names()),
            'trained_at':    obj.get('trained_at'),
            'n_samples':     obj.get('n_samples'),
        }
        log.info('modèle comportement chargé (classes=%s, samples=%s)',
                 _model_meta['classes'], _model_meta['n_samples'])
        return True
    except Exception as e:
        log.exception('échec chargement modèle %s: %s', path, e)
        _model = None
        _model_meta = None
        return False


def is_loaded():
    return _model is not None


def get_meta():
    return dict(_model_meta) if _model_meta else None


def predict(agg_features):
    """Retourne (label, confidence) ou None si pas de modèle / features invalides."""
    if _model is None or agg_features is None:
        return None
    try:
        proba = _model.predict_proba([agg_features])[0]
        idx   = max(range(len(proba)), key=lambda i: proba[i])
        return _model_meta['classes'][idx], float(proba[idx])
    except Exception as e:
        log.warning('predict error: %s', e)
        return None

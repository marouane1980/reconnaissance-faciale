"""Détection de chute robuste — machine à état + signaux d'impact.

Pourquoi c'est mieux que l'approche précédente :
  - Ne dépend plus de la classification "allongé" (qui peut échouer).
  - Détecte l'événement de chute via 4 signaux indépendants :
      1. Vélocité verticale du centre de gravité (descente rapide)
      2. Vélocité d'inclinaison du tronc (basculement brutal)
      3. Variation rapide du ratio hauteur/largeur de la silhouette
      4. Descente rapide du nez (point haut le plus fiable)
  - Machine à état UPRIGHT → FALLING → IMPACT → FALLEN → CONFIRMED
    qui distingue une vraie chute (rapide + reste au sol) d'un coucher
    volontaire (lent) ou d'une assise rapide (s'arrête en hauteur).

Utilisation :
    fd = FallDetector(on_fall=callback)
    for frame: fd.update(ts, lm, frame)
"""

import math
import time
import logging
from collections import deque

log = logging.getLogger('faceid.fall')


# Indices MediaPipe Pose
NOSE       = 0
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28


def _midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _torso_tilt_deg(sh_mid, hip_mid):
    dx = hip_mid[0] - sh_mid[0]
    dy = hip_mid[1] - sh_mid[1]
    return abs(math.degrees(math.atan2(dx, max(dy, 1e-6))))


def _bbox_aspect(lm, vis_thr=0.4):
    """Ratio hauteur/largeur de la silhouette (>1 = vertical, <1 = horizontal)."""
    pts = [(p.x, p.y) for p in lm if getattr(p, 'visibility', 1.0) >= vis_thr]
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    h = max(ys) - min(ys)
    w = max(xs) - min(xs)
    if w < 0.01:
        return None
    return h / w


# ════════════════════════════════════════════
#  États
# ════════════════════════════════════════════

S_UPRIGHT  = 'upright'      # tronc vertical, pas en mouvement de chute
S_FALLING  = 'falling'      # signaux de chute détectés (en cours)
S_IMPACT   = 'impact'       # vient de heurter le sol
S_FALLEN   = 'fallen'       # reste au sol
S_RECOVER  = 'recover'      # se relève


# ════════════════════════════════════════════
#  Détecteur
# ════════════════════════════════════════════

class FallDetector:

    DEFAULTS = {
        # Seuils de détection (tous tunables)
        'min_vel_cog':      0.45,      # vitesse min. de descente du COG (norm/s) pour entrer FALLING
        'min_vel_nose':     0.55,      # vitesse min. de descente du nez (norm/s) — alternative
        'min_tilt_rate':    50.0,      # taux d'inclinaison min. (deg/s) — alternative
        'min_aspect_drop':  0.40,      # baisse rapide du ratio h/w en 1 s
        'fall_max_dur':     2.0,       # durée max DEBOUT → IMPACT pour que ce soit une chute
        'impact_tilt':      45.0,      # tronc déjà bien horizontal pour passer en IMPACT
        'fallen_tilt':      55.0,      # confirmation au sol
        'fallen_min_dur':   1.5,       # rester en IMPACT/FALLEN ce temps minimum → CONFIRMED
        'recovery_tilt':    35.0,      # se relève si tilt revient sous ce seuil
        'cooldown_s':       20.0,      # ne pas re-déclencher avant ce délai
        'window_s':         2.5,       # fenêtre de tracking des signaux
        'min_visibility':   0.4,
    }

    def __init__(self, on_fall=None, params=None):
        self.params = dict(self.DEFAULTS)
        if params:
            self.params.update(params)
        self.on_fall = on_fall
        # Buffer (ts, cog_y, nose_y, tilt, aspect)
        self._track = deque(maxlen=120)
        self._state = S_UPRIGHT
        self._fall_start_ts = None        # ts entrée FALLING
        self._impact_ts     = None        # ts entrée IMPACT
        self._last_alert_ts = 0.0
        self._last_metrics  = {}

    # ── API ──
    def reset(self):
        self._track.clear()
        self._state = S_UPRIGHT
        self._fall_start_ts = None
        self._impact_ts = None

    def state(self):
        return self._state

    def metrics(self):
        return dict(self._last_metrics)

    def update_params(self, params):
        if params:
            self.params.update({k: v for k, v in params.items() if k in self.DEFAULTS})

    def update(self, ts, lm, frame=None, name='Inconnu'):
        """Appelée à chaque frame avec landmarks MediaPipe.
        Retourne True si une chute vient d'être confirmée."""
        p = self.params
        # Visibilité du tronc
        if min(getattr(lm[L_SHO], 'visibility', 1.0),
               getattr(lm[R_SHO], 'visibility', 1.0),
               getattr(lm[L_HIP], 'visibility', 1.0),
               getattr(lm[R_HIP], 'visibility', 1.0)) < p['min_visibility']:
            return False

        sh_mid  = _midpoint(lm[L_SHO], lm[R_SHO])
        hip_mid = _midpoint(lm[L_HIP], lm[R_HIP])
        cog_y   = (sh_mid[1] + hip_mid[1]) / 2.0
        nose_y  = lm[NOSE].y if getattr(lm[NOSE], 'visibility', 1.0) >= p['min_visibility'] else cog_y
        tilt    = _torso_tilt_deg(sh_mid, hip_mid)
        aspect  = _bbox_aspect(lm, p['min_visibility'])

        self._track.append((ts, cog_y, nose_y, tilt, aspect))
        # Élague la fenêtre
        cutoff = ts - p['window_s']
        while self._track and self._track[0][0] < cutoff:
            self._track.popleft()

        # Calcule les signaux sur des sous-fenêtres glissantes
        signals = self._compute_signals(ts)
        self._last_metrics = {
            'state':         self._state,
            'tilt':          round(tilt, 1),
            'cog_y':         round(cog_y, 3),
            'aspect':        round(aspect, 2) if aspect else None,
            **signals,
        }

        confirmed = self._step_state_machine(ts, signals, tilt)
        if confirmed and self.on_fall:
            try:
                self.on_fall({
                    'timestamp': ts,
                    'name':      name,
                    'frame':     frame,
                    'metrics':   dict(self._last_metrics),
                })
            except Exception as e:
                log.exception('on_fall callback error: %s', e)
        return confirmed

    # ── Calcul des signaux ──
    def _compute_signals(self, ts):
        """Sur une fenêtre de 1s : vitesse COG, vitesse nez, taux tilt, chute aspect."""
        win_1s = [r for r in self._track if ts - r[0] <= 1.0]
        if len(win_1s) < 3:
            return {'v_cog': 0.0, 'v_nose': 0.0, 'tilt_rate': 0.0,
                    'aspect_drop': 0.0, 'tilt_max': 0.0}
        t0, c0, n0, _, a0 = win_1s[0]
        t1, c1, n1, _, a1 = win_1s[-1]
        dt = max(t1 - t0, 0.05)

        v_cog       = (c1 - c0) / dt           # > 0 = descente
        v_nose      = (n1 - n0) / dt
        tilts       = [r[3] for r in win_1s]
        tilt_rate   = (tilts[-1] - tilts[0]) / dt
        tilt_max    = max(tilts)
        # Aspect ratio drop : ratio passé > maintenant
        aspects     = [r[4] for r in win_1s if r[4] is not None]
        aspect_drop = (aspects[0] - aspects[-1]) if (len(aspects) >= 2) else 0.0

        return {
            'v_cog':       round(v_cog,     3),
            'v_nose':      round(v_nose,    3),
            'tilt_rate':   round(tilt_rate, 1),
            'aspect_drop': round(aspect_drop, 2),
            'tilt_max':    round(tilt_max,  1),
        }

    # ── Machine à état ──
    def _step_state_machine(self, ts, sig, tilt):
        p = self.params
        confirmed = False

        # Détection d'un mouvement de chute (3 signaux possibles, OR)
        is_falling_signal = (
            sig['v_cog']      >= p['min_vel_cog']  or
            sig['v_nose']     >= p['min_vel_nose'] or
            sig['tilt_rate']  >= p['min_tilt_rate'] or
            sig['aspect_drop'] >= p['min_aspect_drop']
        )

        if self._state == S_UPRIGHT:
            # Filtrage : on ne lance FALLING que si le tronc commence vraiment à basculer
            if is_falling_signal and tilt < p['impact_tilt']:
                self._state = S_FALLING
                self._fall_start_ts = ts
                self._impact_ts = None

        elif self._state == S_FALLING:
            # Trop long sans atteindre IMPACT → faux positif (l'utilisateur s'est juste accroupi)
            if self._fall_start_ts and ts - self._fall_start_ts > p['fall_max_dur']:
                self._state = S_UPRIGHT
                self._fall_start_ts = None
                return False

            # Impact si tronc bien incliné
            if tilt >= p['impact_tilt']:
                self._state = S_IMPACT
                self._impact_ts = ts

        elif self._state == S_IMPACT:
            # Confirmé en chute s'il reste au sol assez longtemps
            if tilt >= p['fallen_tilt']:
                self._state = S_FALLEN
            elif tilt < p['recovery_tilt']:
                # Faux positif : s'est relevé tout de suite
                self._state = S_UPRIGHT
                self._fall_start_ts = None
                self._impact_ts = None

        elif self._state == S_FALLEN:
            if (self._impact_ts and ts - self._impact_ts >= p['fallen_min_dur']
                    and ts - self._last_alert_ts >= p['cooldown_s']):
                # ✓ chute confirmée
                self._last_alert_ts = ts
                confirmed = True
                self._state = S_RECOVER  # ne re-déclenche plus avant que la personne se relève

            elif tilt < p['recovery_tilt']:
                # S'est relevé avant la confirmation → faux positif
                self._state = S_UPRIGHT
                self._fall_start_ts = None
                self._impact_ts = None

        elif self._state == S_RECOVER:
            # Reste muet jusqu'à ce que la personne soit revenue debout
            if tilt < p['recovery_tilt']:
                self._state = S_UPRIGHT
                self._fall_start_ts = None
                self._impact_ts = None

        return confirmed

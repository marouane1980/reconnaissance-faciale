"""Journalisation des apparitions de visages en temps réel."""

import time
import threading
import base64
from datetime import datetime
import cv2

_lock       = threading.Lock()
_log        = {}      # {id: entry_dict}
_log_order  = []      # [id, ...] newest first
_active     = {}      # {name: {id, last_seen_ts}}
_id_counter = 0
GONE_AFTER  = 4.0     # secondes avant de marquer "parti"
MAX_LOG     = 500     # entrées conservées max


def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def update(results, frame=None, analyze_results=None):
    """Appeler à chaque cycle de reconnaissance.
    results = [(x, y, w, h, name), ...]
    """
    global _id_counter
    ts  = time.time()
    now = _now_str()

    # Agrège par nom (plusieurs inconnus → un seul "Inconnu" actif)
    names = {}
    for (x, y, w, h, name) in results:
        if name not in names:
            names[name] = (int(x), int(y), int(w), int(h))

    with _lock:
        # Marque parti les visages qui ont disparu (timeout)
        for name in list(_active.keys()):
            if ts - _active[name]['last_seen_ts'] > GONE_AFTER:
                _log[_active[name]['id']]['status'] = 'gone'
                del _active[name]

        # Crée ou met à jour les entrées actives
        for name, (x, y, w, h) in names.items():
            if name in _active:
                info  = _active[name]
                info['last_seen_ts'] = ts
                entry = _log[info['id']]
                entry['last_seen'] = now
                start = datetime.strptime(entry['first_seen'], '%Y-%m-%d %H:%M:%S')
                entry['duration_s'] = int((datetime.now() - start).total_seconds())
            else:
                _id_counter += 1
                eid   = _id_counter
                photo = None

                # Capture miniature live pour les personnes connues
                if frame is not None and name != 'Inconnu':
                    try:
                        pad = int(max(w, h) * 0.15)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(frame.shape[1], x + w + pad)
                        y2 = min(frame.shape[0], y + h + pad)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            ok, buf = cv2.imencode('.jpg', crop,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 82])
                            if ok and buf is not None:
                                photo = ('data:image/jpeg;base64,'
                                         + base64.b64encode(buf).decode())
                    except Exception:
                        pass

                _log[eid] = {
                    'id':         eid,
                    'name':       name,
                    'photo':      photo,
                    'first_seen': now,
                    'last_seen':  now,
                    'duration_s': 0,
                    'status':     'present',
                    'camera':     'Caméra 1',
                    # Démographie (inconnus, remplie par DeepFace)
                    'age':        None,
                    'age_range':  None,
                    'gender':     None,
                    'face_size':  None,
                }
                _log_order.insert(0, eid)
                _active[name] = {'id': eid, 'last_seen_ts': ts}

                # Limite la taille du journal
                if len(_log_order) > MAX_LOG:
                    old = _log_order.pop()
                    _log.pop(old, None)

        # Attache la démographie DeepFace à l'entrée "Inconnu" active
        if analyze_results and 'Inconnu' in _active:
            eid = _active['Inconnu']['id']
            for ar in analyze_results:
                if not ar.get('error'):
                    e = _log[eid]
                    if ar.get('face'):
                        e['photo'] = ar['face']
                    e.update({
                        'age':       ar.get('age'),
                        'age_range': ar.get('age_range'),
                        'gender':    ar.get('gender'),
                        'face_size': ar.get('face_size'),
                    })
                    break


def update_demographics(analyze_results):
    """Callback appelé par analyzer dès qu'un résultat DeepFace est prêt.
    Met à jour l'entrée Inconnu la plus récente sans démographie, même si elle
    n'est plus active (timeout dépassé).
    """
    if not analyze_results:
        return
    with _lock:
        for ar in analyze_results:
            if ar.get('error'):
                continue
            # Priorité : entrée Inconnu encore active
            target_eid = _active.get('Inconnu', {}).get('id')
            if target_eid is None:
                # Sinon : cherche la plus récente sans démographie
                for eid in _log_order[:20]:
                    e = _log.get(eid)
                    if e and e['name'] == 'Inconnu' and e.get('age') is None:
                        target_eid = eid
                        break
            if target_eid is not None and target_eid in _log:
                e = _log[target_eid]
                if ar.get('face'):
                    e['photo'] = ar['face']
                e.update({
                    'age':       ar.get('age'),
                    'age_range': ar.get('age_range'),
                    'gender':    ar.get('gender'),
                    'face_size': ar.get('face_size'),
                })
            break  # un seul visage inconnu traité par appel


def get_log(limit=500):
    with _lock:
        return [dict(_log[eid]) for eid in _log_order[:limit]]


def get_stats():
    with _lock:
        total     = len(_log_order)
        present   = sum(1 for e in _log.values() if e['status'] == 'present')
        unknowns  = sum(1 for e in _log.values() if e['name'] == 'Inconnu')
        identified = total - unknowns
    return {'total': total, 'present': present,
            'identified': identified, 'unknown': unknowns}


def delete_entries(ids):
    ids_set = set(int(i) for i in ids)
    with _lock:
        for eid in ids_set:
            _log.pop(eid, None)
            for name, info in list(_active.items()):
                if info['id'] == eid:
                    del _active[name]
        _log_order[:] = [eid for eid in _log_order if eid not in ids_set]


def clear():
    global _log, _log_order, _active, _id_counter
    with _lock:
        _log        = {}
        _log_order  = []
        _active     = {}
        _id_counter = 0

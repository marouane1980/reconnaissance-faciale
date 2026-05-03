"""Journalisation des apparitions de visages — persistance SQLite."""

import time
import threading
import base64
import sqlite3
from datetime import datetime
import cv2

_lock      = threading.Lock()
_log       = {}     # {id: entry_dict}  in-memory cache (newest 500)
_log_order = []     # [id, ...] newest first
_active    = {}     # {name: {id, last_seen_ts}}
GONE_AFTER = 4.0
MAX_LOG    = 500

DB_FILE = 'history.db'


# ── Database ──────────────────────────────────────

def _open_db():
    c = sqlite3.connect(DB_FILE, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    return c

_db = _open_db()


def _init_db():
    _db.execute('''CREATE TABLE IF NOT EXISTS tracker_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       TEXT    NOT NULL DEFAULT '',
        photo      TEXT,
        first_seen TEXT,
        last_seen  TEXT,
        duration_s INTEGER DEFAULT 0,
        status     TEXT    DEFAULT 'gone',
        camera     TEXT,
        age        INTEGER,
        age_range  TEXT,
        gender     TEXT,
        face_size  TEXT
    )''')
    # On (re)start, mark any leftover 'present' rows from previous session
    _db.execute("UPDATE tracker_log SET status='gone' WHERE status='present'")
    _db.commit()


def _load_db():
    """Load the most recent MAX_LOG rows from DB into memory on startup."""
    rows = _db.execute(
        'SELECT * FROM tracker_log ORDER BY id DESC LIMIT ?', (MAX_LOG,)
    ).fetchall()
    for row in rows:
        e = {k: row[k] for k in row.keys()}
        _log[e['id']] = e
        _log_order.append(e['id'])   # already DESC → newest first


_init_db()
_load_db()


# ── Helpers ───────────────────────────────────────

def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ── Public API ────────────────────────────────────

def update(results, frame=None, analyze_results=None, cam_label='Caméra 1'):
    """Appeler à chaque cycle de reconnaissance.
    results = [(x, y, w, h, name), ...]
    """
    ts  = time.time()
    now = _now_str()

    names = {}
    for (x, y, w, h, name) in results:
        if name not in names:
            names[name] = (int(x), int(y), int(w), int(h))

    with _lock:
        # Mark gone (timeout)
        for name in list(_active.keys()):
            if ts - _active[name]['last_seen_ts'] > GONE_AFTER:
                eid   = _active[name]['id']
                entry = _log.get(eid)
                if entry:
                    entry['status'] = 'gone'
                    _db.execute(
                        "UPDATE tracker_log SET status='gone', last_seen=?, duration_s=? WHERE id=?",
                        (entry['last_seen'], entry['duration_s'], eid)
                    )
                    _db.commit()
                del _active[name]

        # Create or refresh active entries
        for name, (x, y, w, h) in names.items():
            if name in _active:
                info             = _active[name]
                info['last_seen_ts'] = ts
                entry            = _log[info['id']]
                entry['last_seen'] = now
                start = datetime.strptime(entry['first_seen'], '%Y-%m-%d %H:%M:%S')
                entry['duration_s'] = int((datetime.now() - start).total_seconds())
            else:
                photo = None
                if frame is not None and name != 'Inconnu':
                    try:
                        pad = int(max(w, h) * 0.15)
                        x1 = max(0, x - pad);     y1 = max(0, y - pad)
                        x2 = min(frame.shape[1], x + w + pad)
                        y2 = min(frame.shape[0], y + h + pad)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
                            if ok and buf is not None:
                                photo = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
                    except Exception:
                        pass

                entry = {
                    'name':       name,
                    'photo':      photo,
                    'first_seen': now,
                    'last_seen':  now,
                    'duration_s': 0,
                    'status':     'present',
                    'camera':     cam_label,
                    'age':        None,
                    'age_range':  None,
                    'gender':     None,
                    'face_size':  None,
                }
                cur = _db.execute(
                    '''INSERT INTO tracker_log
                       (name,photo,first_seen,last_seen,duration_s,status,camera,
                        age,age_range,gender,face_size)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                    (entry['name'], entry['photo'], entry['first_seen'], entry['last_seen'],
                     entry['duration_s'], entry['status'], entry['camera'],
                     entry['age'], entry['age_range'], entry['gender'], entry['face_size'])
                )
                _db.commit()
                eid        = cur.lastrowid
                entry['id'] = eid
                _log[eid]  = entry
                _log_order.insert(0, eid)
                _active[name] = {'id': eid, 'last_seen_ts': ts}

                if len(_log_order) > MAX_LOG:
                    old = _log_order.pop()
                    _log.pop(old, None)

        # Attach DeepFace demographics to 'Inconnu' active entry
        if analyze_results and 'Inconnu' in _active:
            eid = _active['Inconnu']['id']
            for ar in analyze_results:
                if not ar.get('error'):
                    e = _log.get(eid)
                    if e:
                        if ar.get('face'):
                            e['photo'] = ar['face']
                        e.update({
                            'age':       ar.get('age'),
                            'age_range': ar.get('age_range'),
                            'gender':    ar.get('gender'),
                            'face_size': ar.get('face_size'),
                        })
                        _db.execute(
                            'UPDATE tracker_log SET photo=?,age=?,age_range=?,gender=?,face_size=? WHERE id=?',
                            (e['photo'], e['age'], e['age_range'], e['gender'], e['face_size'], eid)
                        )
                        _db.commit()
                    break


def update_demographics(analyze_results):
    """Callback from analyzer — update demographics for most recent unknown."""
    if not analyze_results:
        return
    with _lock:
        for ar in analyze_results:
            if ar.get('error'):
                continue
            target_eid = _active.get('Inconnu', {}).get('id')
            if target_eid is None:
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
                _db.execute(
                    'UPDATE tracker_log SET photo=?,age=?,age_range=?,gender=?,face_size=? WHERE id=?',
                    (e['photo'], e['age'], e['age_range'], e['gender'], e['face_size'], target_eid)
                )
                _db.commit()
            break


def get_log(limit=500):
    with _lock:
        return [dict(_log[eid]) for eid in _log_order[:limit] if eid in _log]


def get_stats():
    with _lock:
        total      = len(_log_order)
        present    = sum(1 for e in _log.values() if e['status'] == 'present')
        unknowns   = sum(1 for e in _log.values() if e['name'] == 'Inconnu')
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
        if ids_set:
            _db.execute(
                'DELETE FROM tracker_log WHERE id IN ({})'.format(
                    ','.join('?' * len(ids_set))),
                list(ids_set)
            )
            _db.commit()


def clear():
    global _log, _log_order, _active
    with _lock:
        _log       = {}
        _log_order = []
        _active    = {}
        _db.execute('DELETE FROM tracker_log')
        _db.commit()

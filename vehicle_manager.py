"""Base de données des véhicules enregistrés + historique des plaques détectées."""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime

VEHICLES_FILE = 'vehicles.json'
PLATES_DB     = 'plates.db'

_lock      = threading.Lock()
_log       = {}     # {id: entry}
_log_order = []     # [id] newest first
_active    = {}     # {plate_text: {id, last_seen_ts}}
GONE_AFTER = 12.0
MAX_LOG    = 500


# ── Véhicules enregistrés (JSON) ──────────────────────────────────────────────

def _load_vehicles():
    if not os.path.exists(VEHICLES_FILE):
        return {}
    with open(VEHICLES_FILE, encoding='utf-8') as f:
        return json.load(f)


def _save_vehicles(data):
    with open(VEHICLES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── SQLite historique ─────────────────────────────────────────────────────────

def _open_db():
    c = sqlite3.connect(PLATES_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    return c


_db = _open_db()


def _init_db():
    _db.execute('''CREATE TABLE IF NOT EXISTS plate_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        plate       TEXT    NOT NULL,
        plate_img   TEXT,
        frame_img   TEXT,
        first_seen  TEXT,
        last_seen   TEXT,
        duration_s  INTEGER DEFAULT 0,
        status      TEXT    DEFAULT 'gone',
        camera      TEXT,
        color       TEXT,
        vtype       TEXT,
        matched     INTEGER DEFAULT 0,
        owner       TEXT,
        brand       TEXT,
        model       TEXT
    )''')
    _db.execute("UPDATE plate_log SET status='gone' WHERE status='present'")
    _db.commit()


def _load_db():
    rows = _db.execute(
        'SELECT * FROM plate_log ORDER BY id DESC LIMIT ?', (MAX_LOG,)
    ).fetchall()
    for row in rows:
        e = {k: row[k] for k in row.keys()}
        _log[e['id']] = e
        _log_order.append(e['id'])


_init_db()
_load_db()


def _now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# ── API véhicules enregistrés ─────────────────────────────────────────────────

def get_vehicle(plate):
    return _load_vehicles().get(plate)


def list_vehicles():
    return list(_load_vehicles().values())


def add_vehicle(data):
    plate = data.get('plate', '').strip().upper()
    if not plate:
        return False, 'Plaque obligatoire'
    vehicles = _load_vehicles()
    if plate in vehicles:
        return False, 'Plaque déjà enregistrée'
    vehicles[plate] = {
        'plate':         plate,
        'type':          data.get('type', ''),
        'brand':         data.get('brand', ''),
        'model':         data.get('model', ''),
        'color':         data.get('color', ''),
        'chassis':       data.get('chassis', ''),
        'owner':         data.get('owner', ''),
        'notes':         data.get('notes', ''),
        'registered_at': _now_str(),
    }
    _save_vehicles(vehicles)
    return True, 'ok'


def update_vehicle(plate, data):
    vehicles = _load_vehicles()
    if plate not in vehicles:
        return False, 'Véhicule introuvable'
    for k in ('type', 'brand', 'model', 'color', 'chassis', 'owner', 'notes'):
        if k in data:
            vehicles[plate][k] = data[k]
    new_plate = data.get('plate', plate).strip().upper()
    if new_plate and new_plate != plate:
        entry = vehicles.pop(plate)
        entry['plate'] = new_plate
        vehicles[new_plate] = entry
    _save_vehicles(vehicles)
    return True, 'ok'


def delete_vehicle(plate):
    vehicles = _load_vehicles()
    if plate not in vehicles:
        return False
    del vehicles[plate]
    _save_vehicles(vehicles)
    return True


# ── Suivi des apparitions ─────────────────────────────────────────────────────

def update_sighting(result):
    """Enregistre ou met à jour une apparition de plaque (appelé par plate_recognizer)."""
    plate   = result.get('plate', '')
    ts      = result.get('ts', time.time())
    now     = _now_str()
    vehicle = get_vehicle(plate)
    matched = 1 if vehicle else 0

    with _lock:
        # Timeout des plaques absentes
        for p in list(_active.keys()):
            if ts - _active[p]['last_seen_ts'] > GONE_AFTER:
                eid = _active[p]['id']
                e   = _log.get(eid)
                if e:
                    e['status'] = 'gone'
                    _db.execute(
                        "UPDATE plate_log SET status='gone', last_seen=?, duration_s=? WHERE id=?",
                        (e['last_seen'], e['duration_s'], eid)
                    )
                    _db.commit()
                del _active[p]

        if plate in _active:
            info = _active[plate]
            info['last_seen_ts'] = ts
            e = _log[info['id']]
            e['last_seen'] = now
            start = datetime.strptime(e['first_seen'], '%Y-%m-%d %H:%M:%S')
            e['duration_s'] = int((datetime.now() - start).total_seconds())
        else:
            e = {
                'plate':      plate,
                'plate_img':  result.get('plate_img'),
                'frame_img':  result.get('frame_img'),
                'first_seen': now,
                'last_seen':  now,
                'duration_s': 0,
                'status':     'present',
                'camera':     result.get('camera', ''),
                'color':      result.get('color', ''),
                'vtype':      result.get('vtype') or (vehicle.get('type') if vehicle else ''),
                'matched':    matched,
                'owner':      vehicle.get('owner', '') if vehicle else '',
                'brand':      vehicle.get('brand', '') if vehicle else '',
                'model':      vehicle.get('model', '') if vehicle else '',
            }
            cur = _db.execute(
                '''INSERT INTO plate_log
                   (plate,plate_img,frame_img,first_seen,last_seen,duration_s,status,
                    camera,color,vtype,matched,owner,brand,model)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (e['plate'], e['plate_img'], e['frame_img'], e['first_seen'], e['last_seen'],
                 e['duration_s'], e['status'], e['camera'], e['color'], e['vtype'],
                 e['matched'], e['owner'], e['brand'], e['model'])
            )
            _db.commit()
            eid        = cur.lastrowid
            e['id']    = eid
            _log[eid]  = e
            _log_order.insert(0, eid)
            _active[plate] = {'id': eid, 'last_seen_ts': ts}

            if len(_log_order) > MAX_LOG:
                old = _log_order.pop()
                _log.pop(old, None)


# ── API historique ────────────────────────────────────────────────────────────

def get_history(limit=500):
    with _lock:
        return [dict(_log[eid]) for eid in _log_order[:limit] if eid in _log]


def get_stats():
    with _lock:
        total   = len(_log_order)
        present = sum(1 for e in _log.values() if e['status'] == 'present')
        matched = sum(1 for e in _log.values() if e.get('matched'))
        unknown = total - matched
    return {'total': total, 'present': present, 'matched': matched, 'unknown': unknown}


def delete_entries(ids):
    ids_set = {int(i) for i in ids}
    with _lock:
        for eid in ids_set:
            _log.pop(eid, None)
            for p, info in list(_active.items()):
                if info['id'] == eid:
                    del _active[p]
        _log_order[:] = [eid for eid in _log_order if eid not in ids_set]
        if ids_set:
            _db.execute(
                'DELETE FROM plate_log WHERE id IN ({})'.format(','.join('?' * len(ids_set))),
                list(ids_set)
            )
            _db.commit()


def clear_history():
    global _log, _log_order, _active
    with _lock:
        _log       = {}
        _log_order = []
        _active    = {}
        _db.execute('DELETE FROM plate_log')
        _db.commit()

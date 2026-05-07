import os
import re
import json
import time
import shutil
import base64
import logging
import threading
import functools
from collections import defaultdict, deque
import numpy as np
import cv2
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for, send_file
from werkzeug.security import generate_password_hash, check_password_hash
import analyzer
import tracker
import behavior
import plate_recognizer
import vehicle_manager
from camera_manager import CameraManager, ROOMS, FEATURES

# ── Logging ──
LOG_LEVEL = os.environ.get('FACEID_LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('faceid')

app = Flask(__name__)
os.makedirs("known_faces", exist_ok=True)

# ── Anti brute-force /login ──
_LOGIN_WINDOW   = 60      # secondes
_LOGIN_MAX      = 5       # tentatives ratées par fenêtre
_LOGIN_BAN_SEC  = 300     # ban après dépassement
_login_attempts = defaultdict(lambda: deque())   # ip -> deque[timestamps]
_login_bans     = {}                              # ip -> ban_until_ts
_login_lock     = threading.Lock()

def _login_ip():
    return (request.headers.get('X-Forwarded-For') or request.remote_addr or 'unknown').split(',')[0].strip()

def _login_check_ban(ip):
    with _login_lock:
        until = _login_bans.get(ip, 0)
        if until and until > time.time():
            return int(until - time.time())
        if until:
            _login_bans.pop(ip, None)
    return 0

def _login_record_failure(ip):
    now = time.time()
    with _login_lock:
        dq = _login_attempts[ip]
        dq.append(now)
        while dq and now - dq[0] > _LOGIN_WINDOW:
            dq.popleft()
        if len(dq) >= _LOGIN_MAX:
            _login_bans[ip] = now + _LOGIN_BAN_SEC
            dq.clear()
            log.warning('login banned ip=%s for %ss after %d attempts', ip, _LOGIN_BAN_SEC, _LOGIN_MAX)

def _login_record_success(ip):
    with _login_lock:
        _login_attempts.pop(ip, None)
        _login_bans.pop(ip, None)

# ── Secret key persistant ──
_KEY_FILE = '.secret_key'
if os.path.exists(_KEY_FILE):
    with open(_KEY_FILE) as f:
        app.secret_key = f.read().strip()
else:
    _key = os.urandom(32).hex()
    with open(_KEY_FILE, 'w') as f:
        f.write(_key)
    app.secret_key = _key

# ── Gestion des utilisateurs ──
USERS_FILE = 'users.json'

def _load_users():
    if not os.path.exists(USERS_FILE):
        default = {'admin': {'password': generate_password_hash('admin123'), 'role': 'admin'}}
        with open(USERS_FILE, 'w') as f:
            json.dump(default, f, indent=2)
    with open(USERS_FILE) as f:
        return json.load(f)

def _save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            if request.is_json or request.headers.get('Accept', '').startswith('application/json'):
                return jsonify({'error': 'Non authentifié'}), 401
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            return jsonify({'error': 'Accès réservé à l\'administrateur'}), 403
        return f(*args, **kwargs)
    return decorated

# ── Initialisation caméras & modules ──
analyzer.start()
analyzer.set_on_result_callback(tracker.update_demographics)
behavior.start()
plate_recognizer.start()
plate_recognizer.set_on_result_callback(vehicle_manager.update_sighting)

_cam_mgr = CameraManager()


def _annotate_plates(frame, plate_results):
    """Dessine les boîtes des plaques + texte (avec n° chassis si véhicule connu)."""
    for r in plate_results or []:
        bbox = r.get('bbox') or [0, 0, 0, 0]
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        veh = vehicle_manager.get_vehicle(r.get('plate', ''))
        is_known = bool(veh)
        color = (30, 200, 220) if is_known else (30, 140, 240)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        plate_txt = r.get('plate', '?')
        chassis = (veh or {}).get('chassis', '')
        line1 = plate_txt + (' ✔' if is_known else ' ?')
        line2 = ('VIN ' + chassis) if chassis else ''
        (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1) if line2 else ((0, 0), 0)
        bw = max(tw1, tw2) + 12
        bh = th1 + (th2 + 6 if line2 else 0) + 10
        by = max(y - bh - 4, 2)
        cv2.rectangle(frame, (x, by), (x + bw, by + bh), color, cv2.FILLED)
        cv2.putText(frame, line1, (x + 6, by + th1 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if line2:
            cv2.putText(frame, line2, (x + 6, by + th1 + th2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)


def _annotate(frame, results, beh_results=None, plate_results=None):
    identified = sum(1 for (_, _, _, _, n) in results if n != "Inconnu")
    unknown    = sum(1 for (_, _, _, _, n) in results if n == "Inconnu")
    for (x, y, w, h, name) in results:
        color = (30, 200, 30) if name != "Inconnu" else (30, 30, 210)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        ly = max(y - 8, th + 10)
        cv2.rectangle(frame, (x, ly - th - 8), (x + tw + 10, ly + 2), color, cv2.FILLED)
        cv2.putText(frame, name, (x + 5, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    _annotate_plates(frame, plate_results)
    label = "Identifies: {}  Inconnus: {}".format(identified, unknown)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (8, 8), (lw + 18, lh + 18), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, label, (13, lh + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    if beh_results:
        b = beh_results[0]
        btext = "Posture: {} ({}%)".format(b.get('vid_text', '?'), b.get('confidence', 0))
        (bw, bh), _ = cv2.getTextSize(btext, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        bx = frame.shape[1] - bw - 18
        cv2.rectangle(frame, (bx - 6, 8), (frame.shape[1] - 8, bh + 18), (25, 10, 40), cv2.FILLED)
        cv2.putText(frame, btext, (bx, bh + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 160, 255), 2)
    behavior.draw_landmarks_on(frame)
    return frame


def _generate(cam_id):
    cam = _cam_mgr.get(cam_id, safe=True) or {}
    cam_label = cam.get('label', cam_id)
    while True:
        frame = _cam_mgr.get_frame(cam_id)
        if frame is None:
            time.sleep(0.033)
            continue
        results = _cam_mgr.get_results(cam_id)
        plate_results = plate_recognizer.get_results_for_camera(cam_label)
        _annotate(frame, results, behavior.get_results(), plate_results)
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# ════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════

@app.route('/login', methods=['GET'])
def login_page():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    ip = _login_ip()
    ban_remaining = _login_check_ban(ip)
    if ban_remaining > 0:
        return jsonify({
            'error': 'Trop de tentatives — réessayez dans {}s'.format(ban_remaining),
            'banned_for': ban_remaining,
        }), 429
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    users = _load_users()
    if username in users and check_password_hash(users[username]['password'], password):
        session['user'] = username
        session['role'] = users[username].get('role', 'user')
        _login_record_success(ip)
        log.info('login success user=%s ip=%s', username, ip)
        return jsonify({'success': True})
    _login_record_failure(ip)
    log.warning('login failed user=%s ip=%s', username, ip)
    return jsonify({'error': 'Identifiants incorrects'}), 401

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/me')
@login_required
def me():
    return jsonify({'username': session['user'], 'role': session['role']})

# ── Gestion des utilisateurs (admin) ──
@app.route('/users', methods=['GET'])
@login_required
@admin_required
def list_users():
    users = _load_users()
    return jsonify([{'username': u, 'role': v['role']} for u, v in users.items()])

@app.route('/users', methods=['POST'])
@login_required
@admin_required
def create_user():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'user')
    if not username or not password:
        return jsonify({'error': 'Nom d\'utilisateur et mot de passe requis'}), 400
    users = _load_users()
    if username in users:
        return jsonify({'error': 'Utilisateur déjà existant'}), 409
    users[username] = {'password': generate_password_hash(password), 'role': role}
    _save_users(users)
    return jsonify({'success': True, 'username': username})

@app.route('/users/<username>', methods=['DELETE'])
@login_required
@admin_required
def delete_user(username):
    if username == session['user']:
        return jsonify({'error': 'Impossible de supprimer votre propre compte'}), 400
    users = _load_users()
    if username not in users:
        return jsonify({'error': 'Utilisateur introuvable'}), 404
    del users[username]
    _save_users(users)
    return jsonify({'success': True})

# ════════════════════════════════════════════
#  APP ROUTES (protégées)
# ════════════════════════════════════════════

@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session['user'], role=session['role'])

@app.route('/video_feed')
@login_required
def video_feed():
    cam_id = request.args.get('cam_id') or _cam_mgr.first_id()
    return Response(_generate(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_profile', methods=['POST'])
@login_required
def capture_profile():
    data  = request.get_json()
    name  = (data.get('name') or '').strip()
    step  = int(data.get('step', 1))
    total = int(data.get('total', 5))
    cam_id = data.get('cam_id') or _cam_mgr.first_id()
    if not name:
        return jsonify({'error': 'Nom requis'}), 400
    if not 1 <= step <= total:
        return jsonify({'error': 'Étape invalide'}), 400
    frame = _cam_mgr.get_frame(cam_id)
    if frame is None:
        return jsonify({'error': 'Caméra non disponible'}), 500
    faces = _cam_mgr.detect_faces(cam_id, frame)
    if len(faces) == 0:
        return jsonify({'error': 'Aucun visage détecté — rapprochez-vous'}), 400
    slug = name.lower().replace(' ', '_')
    person_dir = os.path.join('known_faces', slug)
    os.makedirs(person_dir, exist_ok=True)
    cv2.imwrite(os.path.join(person_dir, 'profile_{}.jpg'.format(step)), frame)
    if step == total:
        _cam_mgr.reload_all_faces()
        return jsonify({'success': True, 'done': True, 'name': name})
    return jsonify({'success': True, 'done': False, 'step': step})

@app.route('/faces')
@login_required
def get_faces():
    result = []
    for item in os.listdir('known_faces'):
        item_path = os.path.join('known_faces', item)
        if os.path.isdir(item_path):
            profiles = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            result.append({'name': item.replace('_', ' ').title(), 'profiles': len(profiles)})
        elif item.lower().endswith(('.jpg','.jpeg','.png')):
            result.append({'name': os.path.splitext(item)[0].replace('_',' ').title(), 'profiles': 1})
    return jsonify(sorted(result, key=lambda x: x['name']))

@app.route('/stats')
@login_required
def stats():
    cam_id  = request.args.get('cam_id') or _cam_mgr.first_id()
    results = _cam_mgr.get_results(cam_id) or []
    identified = sum(1 for (_, _, _, _, n) in results if n != "Inconnu")
    unknown    = sum(1 for (_, _, _, _, n) in results if n == "Inconnu")
    return jsonify({'identified': identified, 'unknown': unknown, 'total': len(results)})

@app.route('/threshold', methods=['GET', 'POST'])
@login_required
def threshold():
    if request.method == 'POST':
        try:
            value = float(request.json.get('value', 0.363))
            value = max(0.1, min(0.9, value))
            _cam_mgr.set_threshold(value)
            return jsonify({'success': True, 'threshold': value})
        except (TypeError, ValueError):
            return jsonify({'error': 'Valeur invalide'}), 400
    return jsonify({'threshold': _cam_mgr.get_threshold()})

@app.route('/delete_face/<name>', methods=['DELETE'])
@login_required
def delete_face(name):
    slug   = name.lower().replace(' ', '_')
    folder = os.path.join('known_faces', slug)
    single = folder + '.jpg'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        _cam_mgr.reload_all_faces()
        return jsonify({'success': True})
    if os.path.exists(single):
        os.remove(single)
        _cam_mgr.reload_all_faces()
        return jsonify({'success': True})
    return jsonify({'error': 'Introuvable'}), 404

@app.route('/analyze')
@login_required
def analyze():
    return jsonify({
        'available': analyzer.is_available(),
        'results': analyzer.get_results()
    })

@app.route('/behavior')
@login_required
def behavior_route():
    return jsonify({
        'available': behavior.is_available(),
        'results': behavior.get_results()
    })

@app.route('/behavior/history')
@login_required
def behavior_history():
    return jsonify(behavior.get_history())

@app.route('/behavior/history/clear', methods=['POST'])
@login_required
def behavior_history_clear():
    behavior.clear_history()
    return jsonify({'success': True})

@app.route('/behavior/falls')
@login_required
def behavior_falls():
    return jsonify(behavior.get_fall_history())

@app.route('/behavior/falls/clear', methods=['POST'])
@login_required
def behavior_falls_clear():
    behavior.clear_fall_history()
    return jsonify({'success': True})

@app.route('/behavior/settings', methods=['GET'])
@login_required
def behavior_settings_get():
    return jsonify(behavior.get_settings())

@app.route('/behavior/settings', methods=['POST'])
@login_required
def behavior_settings_post():
    data = request.get_json() or {}
    behavior.apply_settings(data)
    return jsonify({'success': True, 'settings': behavior.get_settings()})

@app.route('/analyze/force_all', methods=['POST'])
@login_required
def analyze_force_all():
    cam_id  = (request.get_json() or {}).get('cam_id') or _cam_mgr.first_id()
    frame   = _cam_mgr.get_frame(cam_id)
    results = _cam_mgr.get_results(cam_id) or []
    if frame is None:
        return jsonify({'error': 'Caméra non disponible'}), 500
    if not results:
        return jsonify({'error': 'Aucun visage dans le cadre'}), 400
    analyzer.submit_all(frame, results)
    return jsonify({'success': True, 'count': len(results)})

@app.route('/analyze/all_results')
@login_required
def analyze_all_results():
    return jsonify({
        'available': analyzer.is_available(),
        'results':   analyzer.get_all_results()
    })

@app.route('/history')
@login_required
def history():
    return jsonify(tracker.get_log())

@app.route('/history/stats')
@login_required
def history_stats():
    return jsonify(tracker.get_stats())

@app.route('/history/delete', methods=['POST'])
@login_required
def history_delete():
    data = request.get_json() or {}
    ids  = data.get('ids', [])
    tracker.delete_entries(ids)
    return jsonify({'success': True, 'deleted': len(ids)})

@app.route('/history/clear', methods=['POST'])
@login_required
def history_clear():
    tracker.clear()
    return jsonify({'success': True})

@app.route('/face_photos/<name>')
@login_required
def face_photos(name):
    """Return all profile photos for a person as base64 data-URLs."""
    slug   = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))
    folder = os.path.join('known_faces', slug)
    photos = []
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    with open(os.path.join(folder, f), 'rb') as fh:
                        photos.append('data:image/jpeg;base64,' + base64.b64encode(fh.read()).decode())
                except Exception:
                    pass
    else:
        single = os.path.join('known_faces', slug + '.jpg')
        if os.path.exists(single):
            try:
                with open(single, 'rb') as fh:
                    photos.append('data:image/jpeg;base64,' + base64.b64encode(fh.read()).decode())
            except Exception:
                pass
    return jsonify(photos)


@app.route('/face_photo/<slug>')
@login_required
def face_photo(slug):
    slug = re.sub(r'[^a-z0-9_]', '', slug.lower())
    folder = os.path.join('known_faces', slug)
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                return send_file(os.path.join(folder, f), mimetype='image/jpeg')
    single = os.path.join('known_faces', slug + '.jpg')
    if os.path.exists(single):
        return send_file(single, mimetype='image/jpeg')
    return '', 404

# ════════════════════════════════════════════
#  CAMERAS ADMIN ROUTES
# ════════════════════════════════════════════

@app.route('/cameras', methods=['GET'])
@login_required
def cameras_list():
    return jsonify(_cam_mgr.list())

@app.route('/cameras', methods=['POST'])
@login_required
@admin_required
def cameras_add():
    data = request.get_json() or {}
    cam_id = _cam_mgr.add(data)
    return jsonify({'success': True, 'id': cam_id})

@app.route('/cameras/<cam_id>', methods=['GET'])
@login_required
def cameras_get(cam_id):
    cam = _cam_mgr.get(cam_id, safe=True)
    if cam is None:
        return jsonify({'error': 'Caméra introuvable'}), 404
    return jsonify(cam)

@app.route('/cameras/<cam_id>', methods=['PUT'])
@login_required
@admin_required
def cameras_update(cam_id):
    data = request.get_json() or {}
    if not _cam_mgr.update(cam_id, data):
        return jsonify({'error': 'Caméra introuvable'}), 404
    return jsonify({'success': True})

@app.route('/cameras/<cam_id>', methods=['DELETE'])
@login_required
@admin_required
def cameras_delete(cam_id):
    if not _cam_mgr.delete(cam_id):
        return jsonify({'error': 'Caméra introuvable'}), 404
    return jsonify({'success': True})

@app.route('/cameras/<cam_id>/toggle', methods=['POST'])
@login_required
@admin_required
def cameras_toggle(cam_id):
    data    = request.get_json() or {}
    enabled = bool(data.get('enabled', True))
    if not _cam_mgr.toggle(cam_id, enabled):
        return jsonify({'error': 'Caméra introuvable'}), 404
    return jsonify({'success': True})

@app.route('/cameras/meta')
@login_required
def cameras_meta():
    return jsonify({'rooms': ROOMS, 'features': FEATURES})


@app.route('/cameras/test', methods=['POST'])
@login_required
@admin_required
def cameras_test():
    """Ouvre une connexion de test sur l'URL fournie et retourne le résultat."""
    data = request.get_json() or {}
    raw  = str(data.get('url', '0')).strip()
    proto = data.get('protocol', 'webcam')
    user  = (data.get('username') or '').strip()
    pwd   = (data.get('password') or '').strip()
    try:
        url = int(raw)
    except ValueError:
        url = raw
        if user and pwd and isinstance(url, str) and url.startswith('rtsp://') and '@' not in url:
            url = 'rtsp://' + user + ':' + pwd + '@' + url[7:]
    cap = None
    try:
        cap = cv2.VideoCapture(url)
        opened = cap.isOpened()
        if not opened:
            return jsonify({'success': False, 'error': "Impossible d'ouvrir la source"}), 200
        ok, frame = cap.read()
        if not ok or frame is None:
            return jsonify({'success': False, 'error': 'Source ouverte mais aucune frame lue'}), 200
        h, w = frame.shape[:2]
        return jsonify({'success': True, 'width': int(w), 'height': int(h), 'protocol': proto})
    except Exception as e:
        log.exception('camera test failed url=%s', url)
        return jsonify({'success': False, 'error': str(e)}), 200
    finally:
        if cap is not None:
            cap.release()


# ════════════════════════════════════════════
#  PLATES & VEHICLES ROUTES
# ════════════════════════════════════════════

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'ocr':    plate_recognizer.is_ready() if hasattr(plate_recognizer, 'is_ready') else plate_recognizer.is_available(),
        'cameras': len(_cam_mgr.all_ids()),
    })


@app.route('/plates/status')
@login_required
def plates_status():
    return jsonify({
        'available': plate_recognizer.is_available(),
        'ready':     plate_recognizer.is_ready() if hasattr(plate_recognizer, 'is_ready') else plate_recognizer.is_available(),
    })


@app.route('/plates/results')
@login_required
def plates_results():
    results = []
    for r in plate_recognizer.get_results():
        item = dict(r)
        veh = vehicle_manager.get_vehicle(item.get('plate', ''))
        if veh:
            item['matched'] = True
            item['chassis'] = veh.get('chassis', '')
            item['owner']   = veh.get('owner', '')
            item['brand']   = veh.get('brand', '')
            item['model']   = veh.get('model', '')
        else:
            item['matched'] = False
        results.append(item)
    return jsonify({
        'available': plate_recognizer.is_available(),
        'results':   results,
    })

@app.route('/plates/history')
@login_required
def plates_history():
    return jsonify(vehicle_manager.get_history())

@app.route('/plates/stats')
@login_required
def plates_stats():
    return jsonify(vehicle_manager.get_stats())

@app.route('/plates/history/delete', methods=['POST'])
@login_required
def plates_history_delete():
    data = request.get_json() or {}
    ids  = data.get('ids', [])
    vehicle_manager.delete_entries(ids)
    return jsonify({'success': True, 'deleted': len(ids)})

@app.route('/plates/history/clear', methods=['POST'])
@login_required
def plates_history_clear():
    vehicle_manager.clear_history()
    return jsonify({'success': True})

@app.route('/vehicles', methods=['GET'])
@login_required
def vehicles_list():
    return jsonify(vehicle_manager.list_vehicles())

@app.route('/vehicles/capture_from_cam', methods=['POST'])
@login_required
def vehicles_capture_from_cam():
    """Récupère la frame courante d'une caméra et lance l'extraction OCR."""
    cam_id = (request.get_json() or {}).get('cam_id') or _cam_mgr.first_id()
    if not cam_id:
        return jsonify({'error': 'Aucune caméra disponible'}), 400
    frame = _cam_mgr.get_frame(cam_id)
    if frame is None:
        return jsonify({'error': 'Caméra non disponible'}), 503
    if not plate_recognizer.is_available():
        return jsonify({'error': 'Module OCR non disponible (easyocr)'}), 503
    info = plate_recognizer.extract_vehicle_info(frame)
    info['matched']  = False
    info['existing'] = None
    if info.get('plate'):
        existing = vehicle_manager.get_vehicle(info['plate'])
        if existing:
            info['matched']  = True
            info['existing'] = existing
    # Snapshot de la frame
    try:
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            info['frame_img'] = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    except Exception:
        pass
    return jsonify({'success': True, 'info': info})


@app.route('/vehicles/extract_from_image', methods=['POST'])
@login_required
def vehicles_extract_from_image():
    """Analyse une image uploadée et retourne plaque/chassis/couleur/type détectés."""
    img_data = None
    if request.files and 'image' in request.files:
        img_data = request.files['image'].read()
    elif request.is_json:
        b64 = (request.get_json() or {}).get('image_b64', '')
        if b64.startswith('data:'):
            b64 = b64.split(',', 1)[-1]
        try:
            img_data = base64.b64decode(b64)
        except Exception:
            img_data = None
    if not img_data:
        return jsonify({'error': 'Image manquante'}), 400
    try:
        arr   = np.frombuffer(img_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        frame = None
    if frame is None:
        return jsonify({'error': 'Image illisible'}), 400
    if not plate_recognizer.is_available():
        return jsonify({'error': 'Module OCR non disponible (easyocr)'}), 503
    info = plate_recognizer.extract_vehicle_info(frame)
    info['matched'] = False
    info['existing'] = None
    if info.get('plate'):
        existing = vehicle_manager.get_vehicle(info['plate'])
        if existing:
            info['matched']  = True
            info['existing'] = existing
    return jsonify({'success': True, 'info': info})


@app.route('/vehicles', methods=['POST'])
@login_required
@admin_required
def vehicles_add():
    data = request.get_json() or {}
    ok, msg = vehicle_manager.add_vehicle(data)
    if ok:
        return jsonify({'success': True})
    return jsonify({'error': msg}), 400

@app.route('/vehicles/<plate>', methods=['PUT'])
@login_required
@admin_required
def vehicles_update(plate):
    data = request.get_json() or {}
    ok, msg = vehicle_manager.update_vehicle(plate.upper(), data)
    if ok:
        return jsonify({'success': True})
    return jsonify({'error': msg}), 404

@app.route('/vehicles/<plate>', methods=['DELETE'])
@login_required
@admin_required
def vehicles_delete(plate):
    if vehicle_manager.delete_vehicle(plate.upper()):
        return jsonify({'success': True})
    return jsonify({'error': 'Véhicule introuvable'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('FACEID_PORT', '5000'))
    log.info('Démarrage Flask sur 0.0.0.0:%d (log level=%s)', port, LOG_LEVEL)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

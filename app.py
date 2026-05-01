import os
import json
import time
import shutil
import threading
import functools
import cv2
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from face_recognizer import FaceRecognizer

app = Flask(__name__)
os.makedirs("known_faces", exist_ok=True)

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

# ── Caméra & reconnaissance ──
_lock = threading.Lock()
_frame = None
_results = []
_recognizer = FaceRecognizer()

def _capture_loop():
    global _frame, _results
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Caméra inaccessible")
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        with _lock:
            _frame = frame.copy()
        idx += 1
        if idx % 4 == 0:
            results = _recognizer.recognize(frame)
            with _lock:
                _results = results

threading.Thread(target=_capture_loop, daemon=True).start()

def _annotate(frame, results):
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
    label = "Identifies: {}  Inconnus: {}".format(identified, unknown)
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (8, 8), (lw + 18, lh + 18), (20, 20, 20), cv2.FILLED)
    cv2.putText(frame, label, (13, lh + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    return frame

def _generate():
    while True:
        with _lock:
            frame = _frame.copy() if _frame is not None else None
            results = list(_results)
        if frame is None:
            time.sleep(0.033)
            continue
        _annotate(frame, results)
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
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    users = _load_users()
    if username in users and check_password_hash(users[username]['password'], password):
        session['user'] = username
        session['role'] = users[username].get('role', 'user')
        return jsonify({'success': True})
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
    return Response(_generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_profile', methods=['POST'])
@login_required
def capture_profile():
    data = request.get_json()
    name  = (data.get('name') or '').strip()
    step  = int(data.get('step', 1))
    total = int(data.get('total', 5))
    if not name:
        return jsonify({'error': 'Nom requis'}), 400
    if not 1 <= step <= total:
        return jsonify({'error': 'Étape invalide'}), 400
    with _lock:
        frame = _frame.copy() if _frame is not None else None
    if frame is None:
        return jsonify({'error': 'Caméra non disponible'}), 500
    faces = _recognizer.detect_faces(frame)
    if len(faces) == 0:
        return jsonify({'error': 'Aucun visage détecté — rapprochez-vous'}), 400
    slug = name.lower().replace(' ', '_')
    person_dir = os.path.join('known_faces', slug)
    os.makedirs(person_dir, exist_ok=True)
    cv2.imwrite(os.path.join(person_dir, 'profile_{}.jpg'.format(step)), frame)
    if step == total:
        _recognizer.load_known_faces()
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
    with _lock:
        results = list(_results)
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
            _recognizer.threshold = value
            return jsonify({'success': True, 'threshold': value})
        except (TypeError, ValueError):
            return jsonify({'error': 'Valeur invalide'}), 400
    return jsonify({'threshold': _recognizer.threshold})

@app.route('/delete_face/<name>', methods=['DELETE'])
@login_required
def delete_face(name):
    slug   = name.lower().replace(' ', '_')
    folder = os.path.join('known_faces', slug)
    single = folder + '.jpg'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        _recognizer.load_known_faces()
        return jsonify({'success': True})
    if os.path.exists(single):
        os.remove(single)
        _recognizer.load_known_faces()
        return jsonify({'success': True})
    return jsonify({'error': 'Introuvable'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

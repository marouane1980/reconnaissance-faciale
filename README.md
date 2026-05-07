# FaceID — Surveillance multi-caméras

Application Flask de surveillance temps-réel combinant **reconnaissance faciale**, **analyse comportementale** (posture & chute) et **lecture de plaques d'immatriculation**.

> Repository : <https://github.com/marouane1980/reconnaissance-faciale>

---

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Démarrage](#démarrage)
- [Variables d'environnement](#variables-denvironnement)
- [Endpoints HTTP](#endpoints-http)
- [Sécurité](#sécurité)
- [Développement](#développement)

---

## Fonctionnalités

### Multi-caméras
- Webcam locale, RTSP, HTTP/MJPEG
- Pool de captures partagées (`_SharedCap`) → un seul flux ouvert par URL
- Activation par caméra : reconnaissance faciale / analyse comportementale / lecture de plaques

### Reconnaissance faciale
- Encodages stockés dans `known_faces/<slug>/profile_*.jpg`
- Wizard de capture multi-photos (5 angles)
- Annotation live : nom, statut Présent/Inconnu, photo de référence
- Historique persistant SQLite (`history.db`) avec apparitions, durées, démographie (âge/sexe estimés)

### Analyse comportementale (MediaPipe Pose)
- Postures détectées : **Debout, Assis(e), Allongé(e), Penché(e), Course, Saut, Grimpe**
- Classification basée sur :
  - inclinaison du tronc en degrés (invariant à la distance caméra)
  - angle des genoux hanche-genou-cheville
  - vote majoritaire sur 7 frames → fin du clignotement
- Détection de chute robuste (4 critères combinés) :
  1. Tronc vertical observé < 1,5 s avant
  2. Vélocité verticale > 0,35 fr/s
  3. Immobilité confirmée 1,5 s après
  4. Cooldown 30 s entre alertes

### Plaques d'immatriculation
- OCR (easyocr) + détection véhicule (YOLOv8)
- Détection couleur dominante (HSV)
- Annotation live : plaque + n° de châssis (VIN) si véhicule connu
- **Capture véhicule depuis image** : OCR sur photo uploadée → préremplit plaque, VIN, couleur, type
- **Capture depuis caméra** : sélection caméra → "Capturer" → extraction live
- Tous les champs restent éditables après capture
- Historique persistant SQLite (`plates.db`)

### Vues & UI
- 3 vues camera : **Grille / Zones / Plan 3D**
- Surveillance live plaques avec flux annoté **sans** reconnaissance faciale (`mode=plates`)
- Panneau "Caméras visibles" : clic = afficher/masquer dans la grille
- Toasts globaux (success/error/warn/info)
- Indicateur de chargement OCR
- Recherche multi-champ dans l'historique
- Bouton "Tester la connexion" sur formulaire caméra

### Rôles utilisateurs
| Rôle | Accès |
|---|---|
| **user** | Dashboard, Reconnaissance faciale (Affichage / Historique), Analyse comportementale (Posture / Historique), Immatriculation (Surveillance live / Historique) |
| **admin** | Tout ce qui précède + menu **Système** (Caméras, Utilisateurs, Reconnaissance faciale, Analyse comportementale, Véhicules) |

---

## Architecture

```
.
├── app.py                  # Application Flask + routes + auth + rate-limit
├── camera_manager.py       # Manager multi-caméras + workers + pool partagé
├── face_recognizer.py      # Reconnaissance via face_recognition (dlib)
├── analyzer.py             # Analyse démographique (âge/sexe)
├── tracker.py              # Tracking apparitions + persistance SQLite
├── behavior.py             # MediaPipe Pose + classification + chute
├── plate_recognizer.py     # OCR easyocr + YOLO + couleur + extraction VIN
├── vehicle_manager.py      # Registre véhicules + historique plaques
├── templates/
│   ├── login.html
│   └── index.html          # SPA monolithique (Tailwind + Lucide)
├── known_faces/            # Photos des personnes (gitignored)
├── captured_faces/         # Frames capturées (gitignored)
├── history.db              # SQLite tracker (gitignored)
├── plates.db               # SQLite plaques (gitignored)
├── users.json              # Comptes (gitignored)
├── cameras.json            # Config caméras (gitignored)
├── vehicles.json           # Registre véhicules (gitignored)
└── requirements.txt
```

---

## Installation

### Prérequis

- **Python 3.10+**
- **CMake + compilateur C++** (pour dlib)
  - Windows : [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Linux : `sudo apt install cmake build-essential`
- ~1 Go d'espace disque pour les modèles ML (easyocr + YOLO + face_recognition)

### Dépendances

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
python setup_models.py    # télécharge YOLOv8n
```

`requirements.txt` contient les versions pinnées : Flask, opencv-contrib-python, numpy, face-recognition, mediapipe, easyocr, ultralytics, Werkzeug, python-docx.

---

## Démarrage

```bash
python app.py
```

L'application écoute sur `http://localhost:5000`.

**Premier login** : `admin` / `admin123` (à changer immédiatement via le menu Système → Utilisateurs).

### Workflow type

1. **Système → Caméras** : ajouter au moins une caméra (webcam = `0`, ou URL RTSP). Cocher les fonctionnalités souhaitées (faciale / comportement / plaques).
2. **Système → Reconnaissance faciale → Ajouter** : enregistrer les personnes connues.
3. **Système → Véhicules** : enregistrer les plaques autorisées (capture image ou caméra).
4. Surveiller via **Dashboard** ou **Surveillance live (plaques)**.

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `FACEID_PORT` | `5000` | Port d'écoute Flask |
| `FACEID_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Endpoints HTTP

### Public
| Méthode | URL | Description |
|---|---|---|
| `GET`  | `/health` | Statut OCR + nombre de caméras (sans auth) |
| `GET`  | `/login` | Page de connexion |
| `POST` | `/login` | Authentification (rate-limited 5/min/IP) |
| `GET`  | `/logout` | Déconnexion |

### Authentifié
| Méthode | URL | Description |
|---|---|---|
| `GET`  | `/me` | Identité de la session |
| `GET`  | `/video_feed?cam_id=X[&mode=plates]` | Flux MJPEG annoté |
| `GET`  | `/faces` / `/face_photos/<name>` | Galerie personnes |
| `POST` | `/capture_profile` | Capture wizard d'enregistrement |
| `GET`  | `/history` / `/history/stats` / `/history/clear` | Apparitions |
| `GET`  | `/behavior` / `/behavior/history` / `/behavior/falls` | Comportement |
| `GET`  | `/plates/results` / `/plates/status` / `/plates/history` | Plaques |
| `POST` | `/vehicles/extract_from_image` | Extraction OCR depuis upload |
| `POST` | `/vehicles/capture_from_cam` | Extraction OCR depuis frame caméra |

### Admin uniquement
| Méthode | URL | Description |
|---|---|---|
| `GET`/`POST`/`PUT`/`DELETE` | `/users` | Gestion comptes |
| `GET`/`POST`/`PUT`/`DELETE` | `/cameras` | Gestion caméras |
| `POST` | `/cameras/test` | Test de connexion (ouvre la source, lit une frame) |
| `POST`/`PUT`/`DELETE` | `/vehicles` | Registre véhicules |

---

## Sécurité

- **Anti brute-force** : 5 échecs `/login` par IP/min déclenchent un ban de 5 min (HTTP 429)
- **Rôles** : décorateurs `@login_required` et `@admin_required`
- **Hash mot de passe** : `werkzeug.security` (pbkdf2 par défaut)
- **Secret key** persistée dans `.secret_key` (gitignoré)
- **Logging structuré** : module `logging` standard, niveau via `FACEID_LOG_LEVEL`
- **Données sensibles gitignorées** : `*.db`, `users.json`, `cameras.json`, `vehicles.json`, `.secret_key`, `known_faces/`, `*.pt`

> ⚠️ Pour un déploiement en production, lancez derrière un reverse-proxy HTTPS (nginx/Caddy) et changez le mot de passe admin par défaut.

---

## Développement

### Vérification syntaxe rapide

```bash
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py','behavior.py','plate_recognizer.py','vehicle_manager.py','camera_manager.py']]; print('OK')"
```

### Synchronisation GitHub (workflow projet)

Après chaque modification :

```bash
git add .
git commit -m "description"
git push origin main
```

### Roadmap

Voir les phases d'amélioration documentées dans les commits :
- ✅ **Phase 1** — Hygiène (gitignore, requirements, rate-limit, logging, toasts, recherche, test caméra)
- ⏳ **Phase 2** — Performance (SSE, skip OCR si pas de mouvement)
- ⏳ **Phase 3** — Alertes (webhook, Telegram), zones d'intérêt
- ⏳ **Phase 4** — DevOps (Docker, CI, tests pytest)
- ⏳ **Phase 5** — Refonte UI (split index.html, mobile, mode clair)
- ⏳ **Phase 6** — IA avancée (DeepSORT, anti-spoofing, masques)

---

## Licence

MIT

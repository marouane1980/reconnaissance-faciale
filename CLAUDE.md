# Instructions pour Claude Code

## Synchronisation GitHub

Après chaque modification du projet (ajout, édition ou suppression de fichier), synchroniser automatiquement avec le repo GitHub :

```bash
git add .
git commit -m "description de la modification"
git push origin main
```

Le repo distant est : <https://github.com/marouane1980/reconnaissance-faciale>

## Aperçu du projet

Application Flask de surveillance multi-caméras combinant **reconnaissance faciale**, **analyse comportementale** et **lecture de plaques d'immatriculation** (voir `README.md`).

### Modules principaux
| Fichier | Rôle |
|---|---|
| `app.py` | Routes Flask, auth, rate-limit, endpoints |
| `camera_manager.py` | Manager multi-caméras + workers + pool partagé |
| `face_recognizer.py` | Reconnaissance via face_recognition (dlib) |
| `analyzer.py` | Analyse démographique (âge/sexe) |
| `tracker.py` | Tracking des apparitions + persistance SQLite |
| `behavior.py` | MediaPipe Pose, classification posture, détection chute |
| `plate_recognizer.py` | OCR easyocr + YOLO + couleur + extraction VIN |
| `vehicle_manager.py` | Registre véhicules + historique plaques |
| `templates/index.html` | SPA monolithique (Tailwind + Lucide) |

### Données persistées (gitignorées)
`history.db`, `plates.db`, `users.json`, `cameras.json`, `vehicles.json`, `.secret_key`, `known_faces/`, `*.pt`

## Conventions

- **Python** : pas de commentaires inutiles, type hints encouragés
- **Logging** : utiliser `logging.getLogger('faceid.<module>')` plutôt que `print()`
- **Routes admin** : décorer avec `@login_required @admin_required`
- **JSON sensibles** ne jamais committer (vérifier `.gitignore` avant `git add .`)
- **Variables d'environnement** : `FACEID_PORT`, `FACEID_LOG_LEVEL`

## Tests rapides

Validation syntaxe Python avant chaque commit :

```bash
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['app.py','behavior.py','plate_recognizer.py','vehicle_manager.py','camera_manager.py']]; print('OK')"
```

## Démarrage local

```bash
python app.py
```

Login par défaut : `admin` / `admin123` (à changer immédiatement).

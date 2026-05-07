# Changelog

Toutes les évolutions notables du projet.

## [Non publié]

### Détection de chute v2 — machine à état + signaux multiples
- Nouveau module `fall_detector.py` avec une **machine à état** :
  `UPRIGHT → FALLING → IMPACT → FALLEN → CONFIRMED → RECOVER`
- Détection **indépendante de la classification de posture** (corrige le cas où
  « allongé » échoue à se déclencher)
- 4 signaux indépendants (OR) déclenchent l'état FALLING :
  - vélocité verticale du COG (centre de gravité)
  - vélocité verticale du nez (point haut très fiable)
  - taux d'inclinaison du tronc en degrés/seconde
  - chute rapide du ratio hauteur/largeur de la silhouette
- Filtrage des faux positifs :
  - faux mouvement court → retour UPRIGHT après 2 s sans impact
  - assise rapide / accroupissement → tilt redescend avant confirmation
  - reste en `RECOVER` jusqu'à ce que la personne se relève (pas de double alerte)
- Confirmation après ≥1,5 s en FALLEN → alerte définitive
- Cooldown abaissé de 30 s à 20 s
- Paramètres tous ajustables via `POST /behavior/falls/params` (admin)
- État live disponible via `GET /behavior/falls/state` (debug overlay)
- Métriques détaillées dans chaque alerte : v_cog, tilt_max, tilt_rate, aspect_drop

### Analyse comportementale — classifieur ML supervisé (Étape A)
- Nouveau module `behavior_classifier.py` : extraction de **28 features par frame**
  (angles, ratios, vitesses, visibilités), agrégation sur fenêtre glissante de 15 frames
  (mean / std / range / velocity → 112 features finales)
- Pipeline d'inférence : si `models/behavior_classifier.pkl` existe, le modèle ML
  prend le pas sur les heuristiques ; sinon fallback automatique
- Réponse `/behavior` enrichie d'un champ `source` : `ml` ou `heuristic`
- API d'enregistrement de clips étiquetés (admin) :
  - `POST /behavior/record/start` (label = debout/assis/allonge/penche/course/saut/grimpe/chute)
  - `POST /behavior/record/stop`
  - `GET  /behavior/record/state` / `list`
  - `DELETE /behavior/record/<fname>`
  - Clips persistés dans `behavior_recordings/*.jsonl` (gitignoré)
- Statut & rechargement modèle :
  - `GET  /behavior/ml/status`
  - `POST /behavior/ml/reload` (admin) — recharge sans redémarrer Flask
- Script `train_behavior.py` : charge les JSONL, entraîne un
  `HistGradientBoostingClassifier`, sauvegarde le pickle dans `models/`
- Mini panneau UI dans Système → Analyse comportementale :
  statut modèle, sélection label, démarrer/arrêter enregistrement, liste des clips,
  bouton « Recharger le modèle »
- `scikit-learn` ajouté aux dépendances optionnelles

### Analyse comportementale (refonte)
- Classification basée sur l'inclinaison du tronc (en degrés) et l'angle des genoux, invariante à la distance caméra
- Nouvelle posture **Penché(e)** (tilt 35–55°) qui évite les anciens faux positifs « Allongé »
- **Vote majoritaire pondéré sur 7 frames** → fin du clignotement de l'UI
- Cadrage haut-buste : si genoux non visibles, retourne **Debout** par défaut au lieu d'« Inconnu »
- **Détection de chute** entièrement repensée (4 critères combinés) :
  1. Tronc vertical (<25°) observé dans la fenêtre de 1,5 s précédente
  2. Vélocité verticale du centre de gravité > 0,35 fr/s
  3. Immobilité confirmée (variance COG faible) pendant 1,5 s après
  4. Cooldown de 30 s entre alertes (corrige le re-déclenchement en boucle)
- Vitesse et tilt enregistrés à chaque alerte de chute
- Logging structuré au lieu de `print`

### Surveillance live (Immatriculation)
- Page restructurée en **Grille / Zones / Plan 3D** identique à la dashboard
- `/video_feed?mode=plates` : flux annoté plaques uniquement (pas de visages)
- Libellé live : « Plaques connues / Inconnues »

### Dashboard
- Sidebar gauche allégée : suppression du switcher de vues et de la liste « Caméras visibles »
- Panneau droit renommé **« Caméras visibles »** : clic = afficher/masquer dans la grille (icône œil ouvert/fermé)

### Menu Système (admin uniquement)
- Visibilité conditionnée à `role === 'admin'` (`<div id="sysWrap" class="hidden">`)
- Contient : Caméras, Utilisateurs, Reconnaissance faciale (page combinée), Analyse comportementale (paramètres), Véhicules
- Page **Reconnaissance faciale** combinée avec onglets : Ajouter / Personnes / Paramètres

### Véhicules
- Nouveau bouton **« Capture automatique »** dans le panneau Véhicules :
  - source = image uploadée (`POST /vehicles/extract_from_image`)
  - source = caméra sélectionnée (`POST /vehicles/capture_from_cam`)
- Extraction OCR : plaque, **n° de châssis (VIN, 17 caractères)**, couleur (HSV), type (YOLO)
- Tous les champs préremplis restent éditables
- Si véhicule déjà connu → bascule en mode édition

### Caméras
- Nouvelle case **« Reconnaissance d'immatriculation »** sur le formulaire caméra
- Bouton **« Tester la connexion »** : ouvre la source, lit une frame, retourne les dimensions
- Icône 🚗 dans la liste des fonctionnalités

### Phase 1 — Hygiène & sécurité
- `.gitignore` complet : `*.db`, `*.db-shm`, `*.db-wal`, `.secret_key`, `users.json`, `cameras.json`, `vehicles.json`, `known_faces/`, `*.pt`, `~$*`, `.DS_Store`, `Thumbs.db`
- `requirements.txt` avec versions pinnées (Flask, opencv, numpy, face-recognition, mediapipe, easyocr, ultralytics, python-docx)
- **Anti brute-force** sur `/login` : 5 tentatives/min/IP puis ban 5 min, retour HTTP 429
- **Logging structuré** (module `logging`, niveau via `FACEID_LOG_LEVEL`)
- `FACEID_PORT` configurable
- Endpoints `/health` et `/plates/status`
- Endpoint admin `/cameras/test`
- `plate_recognizer.is_ready()` (séparé de `is_available()`) pour l'indicateur de chargement OCR

### UX
- **Système de toasts global** (success / error / warn / info)
- **Indicateur de chargement OCR** en haut à droite (apparaît tant qu'easyocr charge)
- **Recherche multi-champ** dans l'historique (nom + caméra + date)
- Remplacement de tous les `alert()` par des toasts
- Police de menu harmonisée sur l'ensemble de la sidebar

## Versions précédentes

Voir `git log` pour l'historique complet des commits.

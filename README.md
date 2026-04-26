# Reconnaissance Faciale en Temps Réel

Projet Python de reconnaissance faciale utilisant **OpenCV** et **face_recognition** (dlib).

## Fonctionnalités

- Détection de visages en temps réel via webcam
- Reconnaissance des visages connus (stockés dans `known_faces/`)
- Affichage du nom sous chaque visage détecté
- Capture de frames avec la touche `S`
- Ajout de nouveaux visages connus via `add_face.py`

## Structure

```
.
├── main.py              # Lancement de la reconnaissance en temps réel
├── add_face.py          # Enregistrer un nouveau visage connu
├── face_recognizer.py   # Module de reconnaissance (face_recognition)
├── face_detector.py     # Module de détection (OpenCV Haar cascade)
├── known_faces/         # Images des personnes connues (prénom_nom.jpg)
├── captured_faces/      # Frames capturées pendant l'exécution
└── requirements.txt
```

## Installation

### Prérequis

- Python 3.8+
- CMake et un compilateur C++ (nécessaire pour `dlib`)  
  - Windows : [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Linux : `sudo apt install cmake build-essential`

### Dépendances Python

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Ajouter un visage connu

**Depuis la webcam :**
```bash
python add_face.py --name "Prénom Nom"
```
Appuyez sur `ESPACE` pour capturer, `Q` pour annuler.

**Depuis une image existante :**
```bash
python add_face.py --name "Prénom Nom" --source chemin/vers/photo.jpg
```

Les images sont sauvegardées dans `known_faces/prenom_nom.jpg`.

### 2. Lancer la reconnaissance

```bash
python main.py
```

| Touche | Action                            |
|--------|-----------------------------------|
| `S`    | Sauvegarder le frame courant      |
| `Q`    | Quitter                           |

## Paramètres

| Paramètre | Fichier | Description |
|-----------|---------|-------------|
| `CAMERA_INDEX` | `main.py` | Index de la caméra (0 = défaut) |
| `PROCESS_EVERY_N_FRAMES` | `main.py` | Fréquence de traitement (performance) |
| `tolerance` | `face_recognizer.py` | Seuil de reconnaissance (0.4 strict → 0.65 souple) |

## Licence

MIT

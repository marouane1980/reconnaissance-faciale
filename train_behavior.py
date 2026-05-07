"""Entraîne un classifieur de postures à partir des clips JSONL enregistrés.

Usage :
    python train_behavior.py
    python train_behavior.py --window 15 --recordings behavior_recordings
    python train_behavior.py --report             # affiche aussi le rapport de classification

Sortie : models/behavior_classifier.pkl

Le modèle peut être rechargé en ligne via POST /behavior/ml/reload.
"""

import os
import json
import glob
import time
import pickle
import argparse
import logging
from collections import Counter, defaultdict

import behavior_classifier as bc

log = logging.getLogger('train_behavior')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def load_clips(directory):
    """Retourne {label: [[features_par_frame, ...], ...]} (liste de clips par label)."""
    clips = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
        frames = []
        label = None
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if label is None:
                    label = d.get('label')
                feats = d.get('features')
                if feats and len(feats) == len(bc.FRAME_FEATURE_NAMES):
                    frames.append(feats)
        if label and len(frames) >= 5:
            clips[label].append(frames)
            log.info('  %s : %s (%d frames)', os.path.basename(path), label, len(frames))
    return clips


def build_dataset(clips, window=15, stride=5):
    """Convertit les clips en (X, y) avec une fenêtre glissante."""
    X, y = [], []
    for label, sessions in clips.items():
        for frames in sessions:
            if len(frames) < max(5, window // 2):
                continue
            buf = bc.FeatureBuffer(window=window)
            for i, f in enumerate(frames):
                buf.push(i / 10.0, f)            # ts fictif (10 fps), n'impacte que la vitesse
                if buf.is_ready() and (i % stride == 0):
                    agg = buf.aggregate()
                    if agg is not None:
                        X.append(agg)
                        y.append(label)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recordings', default='behavior_recordings')
    ap.add_argument('--out',        default=os.path.join('models', 'behavior_classifier.pkl'))
    ap.add_argument('--window',     type=int, default=15)
    ap.add_argument('--stride',     type=int, default=5)
    ap.add_argument('--report',     action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(args.recordings):
        log.error('dossier introuvable : %s', args.recordings)
        return 1

    log.info('chargement des clips depuis %s', args.recordings)
    clips = load_clips(args.recordings)
    if not clips:
        log.error('aucun clip valide trouvé')
        return 1

    log.info('résumé clips : %s', {k: len(v) for k, v in clips.items()})

    X, y = build_dataset(clips, window=args.window, stride=args.stride)
    log.info('dataset : %d échantillons, %d features, classes=%s',
             len(X), len(X[0]) if X else 0, dict(Counter(y)))
    if len(set(y)) < 2:
        log.error('au moins 2 classes différentes requises pour entraîner')
        return 1

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        log.error('scikit-learn requis : pip install scikit-learn')
        return 1

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.08, random_state=42)
    model.fit(Xtr, ytr)
    score = model.score(Xte, yte)
    log.info('précision test : %.3f', score)

    if args.report:
        ypred = model.predict(Xte)
        log.info('\n%s', classification_report(yte, ypred))
        log.info('matrice de confusion :\n%s\nclasses=%s',
                 confusion_matrix(yte, ypred), list(model.classes_))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    payload = {
        'model':         model,
        'classes':       list(model.classes_),
        'feature_names': bc.aggregated_feature_names(),
        'trained_at':    int(time.time()),
        'n_samples':     len(X),
        'window':        args.window,
    }
    with open(args.out, 'wb') as f:
        pickle.dump(payload, f)
    log.info('modèle sauvegardé → %s', args.out)
    log.info('Pour activer en live : POST /behavior/ml/reload (admin)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

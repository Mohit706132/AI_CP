# -*- coding: utf-8 -*-
"""Model versioning manager: backup, staging, comparison, and promotion."""
import shutil
import json
import joblib
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = ROOT / 'uv_module' / 'models'
DEFAULT_DATA_DIR = ROOT / 'uv_module' / 'data'

MODEL_ARTIFACTS = {
    'hirda': {
        'subdir': 'Model',
        'files': ['best_model.pkl', 'label_encoder.pkl', 'pca.pkl', 'pipeline_meta.pkl', 'ratio_transformer.pkl', 'scaler.pkl'],
    },
    'rauwolfia': {
        'subdir': '',
        'files': ['svm_uv.pkl', 'scaler_uv.pkl', 'pca_uv.pkl', 'imputer_uv.pkl', 'reference_standard_uv.json', 'metadata_uv.json'],
    },
    'embelia': {
        'subdir': '',
        'files': [],
        'data_files': ['Embelia_UV_new.csv'],
    },
    'plant_detector': {
        'subdir': '',
        'files': ['plant_detector.pkl'],
    },
}


def _resolve_dirs(models_dir, data_dir):
    m = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return m, d


def _prod_dir(model_key, models_dir):
    m, _ = _resolve_dirs(models_dir, None)
    sub = MODEL_ARTIFACTS[model_key]['subdir']
    return m / sub if sub else m


def _staging_dir(model_key, models_dir):
    m, _ = _resolve_dirs(models_dir, None)
    return m.parent / 'models_staging' / model_key


def _backup_dir(model_key, models_dir):
    m, _ = _resolve_dirs(models_dir, None)
    return m.parent / 'models_backup' / model_key


def backup_production_models(model_key, models_dir=None, data_dir=None):
    """Copy current production model files into models_backup/<key>/."""
    models_dir, data_dir = _resolve_dirs(models_dir, data_dir)
    info = MODEL_ARTIFACTS[model_key]
    prod = _prod_dir(model_key, models_dir)
    backup = _backup_dir(model_key, models_dir)
    backup.mkdir(parents=True, exist_ok=True)

    for fname in info['files']:
        src = prod / fname
        if src.exists():
            shutil.copy2(src, backup / fname)

    if data_dir and 'data_files' in info:
        for fname in info['data_files']:
            src = Path(data_dir) / fname
            if src.exists():
                shutil.copy2(src, backup / fname)

    meta_path = backup / '_backup_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'model_key': model_key}, f)

    return backup


def get_staging_dir(model_key, models_dir=None):
    """Return the staging directory for a model key, creating it if needed."""
    models_dir, _ = _resolve_dirs(models_dir, None)
    staging = _staging_dir(model_key, models_dir)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def get_production_accuracy(model_key, models_dir=None, data_dir=None):
    """Read the stored CV accuracy from the current production model metadata."""
    models_dir, data_dir = _resolve_dirs(models_dir, data_dir)
    try:
        if model_key == 'hirda':
            meta = joblib.load(Path(models_dir) / 'Model' / 'pipeline_meta.pkl')
            return meta.get('cv_accuracy', meta.get('loocv_accuracy', 0.0)) * 100 if meta.get('loocv_accuracy') else meta.get('cv_accuracy', 0.0)
        elif model_key == 'rauwolfia':
            with open(Path(models_dir) / 'metadata_uv.json', 'r') as f:
                meta = json.load(f)
            return meta.get('cv_accuracy', 0.0)
        elif model_key == 'embelia':
            return 88.0
        elif model_key == 'plant_detector':
            art = joblib.load(Path(models_dir) / 'plant_detector.pkl')
            return art.get('cv_accuracy', 0.0) if isinstance(art, dict) else 0.0
    except Exception:
        return 0.0
    return 0.0


def promote_staging(model_key, models_dir=None, data_dir=None):
    """Copy staging artifacts over production artifacts."""
    models_dir, data_dir = _resolve_dirs(models_dir, data_dir)
    info = MODEL_ARTIFACTS[model_key]
    staging = _staging_dir(model_key, models_dir)
    prod = _prod_dir(model_key, models_dir)
    prod.mkdir(parents=True, exist_ok=True)

    for fname in info['files']:
        src = staging / fname
        if src.exists():
            shutil.copy2(src, prod / fname)

    if data_dir and 'data_files' in info:
        for fname in info['data_files']:
            src = staging / fname
            if src.exists():
                shutil.copy2(src, Path(data_dir) / fname)


def rollback_from_backup(model_key, models_dir=None, data_dir=None):
    """Restore production artifacts from backup."""
    models_dir, data_dir = _resolve_dirs(models_dir, data_dir)
    info = MODEL_ARTIFACTS[model_key]
    backup = _backup_dir(model_key, models_dir)
    prod = _prod_dir(model_key, models_dir)

    if not backup.exists():
        raise FileNotFoundError(f'No backup found for {model_key}')

    for fname in info['files']:
        src = backup / fname
        if src.exists():
            shutil.copy2(src, prod / fname)

    if data_dir and 'data_files' in info:
        for fname in info['data_files']:
            src = backup / fname
            if src.exists():
                shutil.copy2(src, Path(data_dir) / fname)


def compare_and_decide(model_key, new_cv_accuracy, models_dir=None, data_dir=None):
    """Compare new model accuracy vs production and return decision dict."""
    models_dir, data_dir = _resolve_dirs(models_dir, data_dir)
    old_acc = get_production_accuracy(model_key, models_dir, data_dir)
    improved = new_cv_accuracy >= old_acc
    decision = 'PROMOTE' if improved else 'REJECT'
    reason = (
        f"New model accuracy ({new_cv_accuracy:.2f}%) exceeds or equals production ({old_acc:.2f}%)."
        if improved else
        f"New model accuracy ({new_cv_accuracy:.2f}%) is lower than production ({old_acc:.2f}%)."
    )

    return {
        'old_accuracy': old_acc,
        'new_accuracy': new_cv_accuracy,
        'delta': new_cv_accuracy - old_acc,
        'improved': improved,
        'decision': decision,
        'recommendation': decision,
        'reason': reason
    }


def cleanup_staging(model_key, models_dir=None):
    """Remove staging directory for a model key."""
    models_dir, _ = _resolve_dirs(models_dir, None)
    staging = _staging_dir(model_key, models_dir)
    if staging.exists():
        shutil.rmtree(staging)

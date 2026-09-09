# -*- coding: utf-8 -*-
"""Model versioning manager: backup, staging, comparison, and promotion."""
import shutil
import json
import joblib
from pathlib import Path
from datetime import datetime

# Which files belong to each model key in the production directory
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


def _prod_dir(model_key, models_dir):
    sub = MODEL_ARTIFACTS[model_key]['subdir']
    return Path(models_dir) / sub if sub else Path(models_dir)


def _staging_dir(model_key, models_dir):
    return Path(models_dir).parent / 'models_staging' / model_key


def _backup_dir(model_key, models_dir):
    return Path(models_dir).parent / 'models_backup' / model_key


def backup_production_models(model_key, models_dir, data_dir=None):
    """Copy current production model files into models_backup/<key>/."""
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


def get_staging_dir(model_key, models_dir):
    """Return the staging directory for a model key, creating it if needed."""
    staging = _staging_dir(model_key, models_dir)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def get_production_accuracy(model_key, models_dir, data_dir=None):
    """Read the stored CV accuracy from the current production model metadata."""
    try:
        if model_key == 'hirda':
            meta = joblib.load(Path(models_dir) / 'Model' / 'pipeline_meta.pkl')
            return meta.get('cv_accuracy', meta.get('loocv_accuracy', 0.0)) * 100 if meta.get('loocv_accuracy') else meta.get('cv_accuracy', 0.0)
        elif model_key == 'rauwolfia':
            with open(Path(models_dir) / 'metadata_uv.json', 'r') as f:
                meta = json.load(f)
            return meta.get('cv_accuracy', 0.0)
        elif model_key == 'embelia':
            return 0.0
        elif model_key == 'plant_detector':
            art = joblib.load(Path(models_dir) / 'plant_detector.pkl')
            return art.get('cv_accuracy', 0.0) if isinstance(art, dict) else 0.0
    except Exception:
        return 0.0
    return 0.0


def promote_staging(model_key, models_dir, data_dir=None):
    """Copy staging artifacts over production artifacts."""
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


def rollback_from_backup(model_key, models_dir, data_dir=None):
    """Restore production artifacts from backup."""
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


def compare_and_decide(model_key, new_cv_accuracy, models_dir, data_dir=None):
    """Compare new model accuracy vs production and return decision dict."""
    old_acc = get_production_accuracy(model_key, models_dir, data_dir)
    improved = new_cv_accuracy >= old_acc

    return {
        'old_accuracy': old_acc,
        'new_accuracy': new_cv_accuracy,
        'delta': new_cv_accuracy - old_acc,
        'improved': improved,
        'recommendation': 'PROMOTE' if improved else 'REJECT',
    }


def cleanup_staging(model_key, models_dir):
    """Remove staging directory for a model key."""
    staging = _staging_dir(model_key, models_dir)
    if staging.exists():
        shutil.rmtree(staging)

"""Learned plant-level UV detector for Auto-Detect routing."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DEFAULT_GRID = np.arange(220.0, 801.0, 2.0)


def _numeric_spectrum(frame: pd.DataFrame):
    columns = []
    for column in frame.columns:
        try:
            columns.append((str(column), float(column)))
        except (TypeError, ValueError):
            continue
    if len(columns) < 50:
        raise ValueError("The uploaded file does not contain enough numeric wavelength columns.")
    columns.sort(key=lambda item: item[1])
    names = [name for name, _ in columns]
    wavelengths = np.asarray([value for _, value in columns], dtype=float)
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The uploaded UV spectrum contains blank or non-numeric values.")
    return wavelengths, values


def prepare_uv(frame: pd.DataFrame, grid=DEFAULT_GRID) -> np.ndarray:
    """Interpolate spectra to the common grid and apply row-wise SNV."""
    wavelengths, values = _numeric_spectrum(frame)
    if wavelengths[0] > grid[0] or wavelengths[-1] < grid[-1]:
        raise ValueError(f"UV spectrum must cover at least {int(grid[0])}-{int(grid[-1])} nm.")
    prepared = np.vstack([np.interp(grid, wavelengths, row) for row in values])
    means = prepared.mean(axis=1, keepdims=True)
    stds = prepared.std(axis=1, keepdims=True)
    return (prepared - means) / (stds + 1e-8)


def load_detector(models_dir: Path):
    artifact_path = Path(models_dir) / "plant_detector.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError("Plant detector model is missing. Run train_plant_detector.py first.")
    return joblib.load(artifact_path)


def detect_plant(frame: pd.DataFrame, models_dir: Path):
    artifact = load_detector(models_dir)
    x = prepare_uv(frame, np.asarray(artifact["grid"], dtype=float))
    model = artifact["model"]
    probabilities = model.predict_proba(x)
    class_names = np.asarray(model.classes_)
    best_indices = np.argmax(probabilities, axis=1)
    best_scores = probabilities[np.arange(len(x)), best_indices]
    threshold = float(artifact.get("confidence_threshold", 0.60))
    labels = np.where(best_scores >= threshold, class_names[best_indices], "UNKNOWN / LOW CONFIDENCE")
    details = [
        {str(label): float(score) for label, score in zip(class_names, row)}
        for row in probabilities
    ]
    return labels, best_scores, details, x

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import PhytochemicalRatioTransformer

EXPECTED_WAVELENGTHS = np.arange(220, 802, 2)
EXPECTED_FEATURE_COUNT = len(EXPECTED_WAVELENGTHS)

def load_hirda_artifacts(models_dir):
    # Already mapped in utils.py but mapping here just in case
    sys.modules['__main__'].PhytochemicalRatioTransformer = PhytochemicalRatioTransformer
    
    scaler = joblib.load(models_dir / "Model/scaler.pkl")
    pca = joblib.load(models_dir / "Model/pca.pkl") if (models_dir / "Model/pca.pkl").exists() else None
    model = joblib.load(models_dir / "Model/best_model.pkl")
    encoder = joblib.load(models_dir / "Model/label_encoder.pkl")
    ratio_trans = joblib.load(models_dir / "Model/ratio_transformer.pkl") if (models_dir / "Model/ratio_transformer.pkl").exists() else None
    return scaler, pca, model, encoder, ratio_trans

def validate_hirda_df(frame: pd.DataFrame):
    clean = frame.copy()
    clean.columns = [str(column).strip() for column in clean.columns]
    
    # Try to detect label col
    label_col = None
    if clean.shape[1] == EXPECTED_FEATURE_COUNT + 1:
        first_vals = clean.iloc[:, 0].dropna().astype(str)
        num_count = sum(1 for v in first_vals.head(5) if v.replace('.','').isdigit())
        if num_count == 0:
            label_col = clean.columns[0]
            
    if label_col:
        feature_frame = clean.iloc[:, 1:].copy()
        actual_labels = clean.iloc[:, 0].astype(str)
    else:
        feature_frame = clean.copy()
        actual_labels = None
        
    if feature_frame.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Hirda Engine expected {EXPECTED_FEATURE_COUNT} wavelength columns, found {feature_frame.shape[1]}.")
        
    expected_cols = [str(int(wv)) for wv in EXPECTED_WAVELENGTHS]
    ordered_frame = feature_frame.loc[:, expected_cols].apply(pd.to_numeric, errors="coerce")
    
    if ordered_frame.isna().any().any():
        raise ValueError("The uploaded CSV contains blank or non-numeric values in the wavelength columns.")
        
    return ordered_frame, actual_labels, label_col

def process_hirda_batch(sample_df, models_dir, filename="sample.csv"):
    ordered_frame, actual_labels, label_col = validate_hirda_df(sample_df)
    scaler, pca, model, encoder, ratio_trans = load_hirda_artifacts(models_dir)
    
    if ratio_trans is not None:
        x_feats = ratio_trans.transform(ordered_frame)
    else:
        x_feats = ordered_frame.to_numpy(dtype=float)
        
    x_scaled = scaler.transform(x_feats)
    
    if pca is not None and hasattr(model, "n_features_in_") and model.n_features_in_ == pca.n_components_:
        x_input = pca.transform(x_scaled)
    else:
        x_input = x_scaled
        
    raw_preds = model.predict(x_input)
    if isinstance(raw_preds[0], (int, np.integer)):
        predicted_labels = encoder.inverse_transform(raw_preds.astype(int))
    else:
        predicted_labels = raw_preds.astype(str)
        
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_input)
        
    results = []
    classes_list = list(getattr(model, "classes_", encoder.classes_))
    
    for i in range(len(ordered_frame)):
        pred = predicted_labels[i]
        true_lbl = actual_labels.iloc[i] if actual_labels is not None else None
        conf_dict = {}
        max_conf = 0.0
        
        if probabilities is not None:
            max_conf = float(np.max(probabilities[i]))
            for j, cls_name in enumerate(classes_list):
                conf_dict[str(cls_name)] = float(probabilities[i][j])
                
        results.append({
            "sample_name": f"{filename} (Row {i+1})",
            "prediction": pred,
            "true_label": true_lbl,
            "adul_status": "AUTHENTIC" if max_conf >= 0.45 else "LOW CONFIDENCE - Verify",
            "purity": round(max_conf * 100, 2),
            "confidence": conf_dict,
            "raw_spectrum": ordered_frame.iloc[i].values.tolist(),
            "modality": "UV"
        })
        
    return results, "UV"

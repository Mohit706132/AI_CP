import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from .utils import apply_snv, detect_modality

def load_rauwolfia_assets(modality, models_dir):
    suffix = modality.lower()
    svm = joblib.load(models_dir / f"svm_{suffix}.pkl")
    scaler = joblib.load(models_dir / f"scaler_{suffix}.pkl")
    pca = joblib.load(models_dir / f"pca_{suffix}.pkl")
    imputer = joblib.load(models_dir / f"imputer_{suffix}.pkl")
    
    # Patch for sklearn version mismatch (1.7 -> 1.8)
    if not hasattr(imputer, "_fill_dtype"):
        imputer._fill_dtype = np.dtype('float64')
        
    with open(models_dir / f"reference_standard_{suffix}.json", "r") as f:
        ref_std = pd.Series(json.load(f))
        
    with open(models_dir / f"metadata_{suffix}.json", "r") as f:
        pca_meta = json.load(f)
        pca_meta['inv_cov'] = np.array(pca_meta['inv_cov'])
            
    return svm, scaler, pca, imputer, ref_std, pca_meta

def process_rauwolfia_batch(sample_df, models_dir, filename="sample.csv"):
    modality = detect_modality(sample_df)
    if modality is None:
        raise ValueError("Could not detect spectral modality from column names.")
    
    svm_model, scaler, pca_model, imputer, reference_std, pca_meta = load_rauwolfia_assets(modality, models_dir)
    expected_columns = pca_meta["feature_columns"]
    
    wvs = []
    for c in sample_df.columns:
        try:
            val = float(c)
            if modality == 'UV' and 200 <= val <= 1200:
                wvs.append((c, val))
            elif modality == 'FTIR' and 400 <= val <= 4000:
                wvs.append((c, val))
        except:
            pass
            
    if len(wvs) < 50:
        raise ValueError(f"Insufficient pure {modality} spectral features ({len(wvs)}).")
        
    wvs.sort(key=lambda x: x[1])
    orig_names = [x[0] for x in wvs]
    orig_vals = np.array([x[1] for x in wvs])
    
    y_old = sample_df[orig_names].values.astype(float)
    target_wvs = np.array(expected_columns).astype(float)
    interp_vals = []
    for row_y in y_old:
        interp_vals.append(np.interp(target_wvs, orig_vals, row_y, left=np.nan, right=np.nan))
        
    processed_df = pd.DataFrame(interp_vals, columns=expected_columns)
    processed_imp_vals = imputer.transform(processed_df.values)
    processed_imp_df = pd.DataFrame(processed_imp_vals, columns=expected_columns)
    
    processed_snv = apply_snv(processed_imp_df)
    X_snv_df = pd.DataFrame(processed_snv, columns=expected_columns)
    
    X_scaled_np = scaler.transform(X_snv_df)
    X_scaled_df = pd.DataFrame(X_scaled_np, columns=expected_columns)
    
    X_pca = pca_model.transform(X_scaled_df)
    X_recon = pca_model.inverse_transform(X_pca)
    
    n_comp = pca_meta.get('n_components', X_pca.shape[1])
    t2_thresh = pca_meta.get('t2_threshold', 99999.0)
    q_thresh = pca_meta.get('q_thresh', 99999.0) if 'q_thresh' in pca_meta else pca_meta.get('q_threshold', 99999.0)
    inv_cov = np.array(pca_meta.get('inv_cov', np.eye(X_pca.shape[1])))

    t2_scores = np.array([s @ inv_cov @ s for s in X_pca])
    q_scores = np.sum((X_scaled_np - X_recon) ** 2, axis=1)

    pca_cols = [f"PC{i+1}" for i in range(n_comp)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_cols)
    predictions = svm_model.predict(X_pca_df)
    probabilities = svm_model.predict_proba(X_pca_df)

    results = []
    for idx in range(len(sample_df)):
        t2 = t2_scores[idx]
        q = q_scores[idx]

        if t2 > t2_thresh and q > q_thresh:
            prediction = "UNKNOWN / FOREIGN"
            confidence = {}
            status = "REJECTED - Unknown/Foreign Plant"
            purity_pct = 0.0
        else:
            prediction = predictions[idx]
            confidence = dict(zip(svm_model.classes_, probabilities[idx]))
            
            if "FOREIGN" in prediction:
                status = "REJECTED - Foreign Plant Identified"
                purity_pct = 0.0
            else:
                pred_conf = confidence[prediction]
                if pred_conf >= 0.70:
                    status = f"AUTHENTIC ({prediction})"
                elif pred_conf >= 0.50:
                    status = f"AUTHENTIC WITH SPECTRAL VARIATION ({prediction})"
                else:
                    status = "SUSPICIOUS - Low confidence"
                purity_pct = round(max(probabilities[idx]) * 100, 2)
                
        true_label = sample_df.iloc[idx]['label'] if 'label' in sample_df.columns else None
        
        results.append({
            "sample_name": f"{filename} (Row {idx+1})",
            "prediction": prediction,
            "true_label": true_label,
            "adul_status": status,
            "purity": purity_pct,
            "confidence": confidence,
            "raw_spectrum": processed_imp_df.iloc[idx].values.tolist(), # Convert to list for easier JSON/DB storage
            "t2_score": float(t2),
            "q_score": float(q),
            "modality": modality
        })
        
    return results, modality

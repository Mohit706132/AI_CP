import pandas as pd
import numpy as np
from pathlib import Path

def authenticate_sample(sample, X_auth, X_adul):
    d_sample = np.diff(sample)
    d_auth = np.diff(X_auth)
    d_adul = np.diff(X_adul)
    
    sims_auth = [np.dot(d_sample, a) / (np.linalg.norm(d_sample) * np.linalg.norm(a) + 1e-9) for a in d_auth]
    sims_adul = [np.dot(d_sample, a) / (np.linalg.norm(d_sample) * np.linalg.norm(a) + 1e-9) for a in d_adul]
    
    max_sim_auth = max(sims_auth) if sims_auth else 0
    max_sim_adul = max(sims_adul) if sims_adul else 0
    
    is_authentic = (max_sim_auth > 0.88) and (max_sim_auth >= max_sim_adul)
    return round(max_sim_auth * 100, 2), is_authentic

def process_embelia_batch(sample_df, data_dir, filename="sample.csv"):
    # Load library standard automatically
    lib_path = Path(data_dir) / "Embelia_UV.csv"
    if not lib_path.exists():
        raise FileNotFoundError("Embelia library standard file not found in data directory.")
        
    df_train = pd.read_csv(lib_path)
    df_test = sample_df
    
    t_vals = set([int(c) for c in df_train.columns[1:] if str(c).isdigit()])
    s_vals = set([int(c) for c in df_test.columns[1:] if str(c).isdigit()])
    common_cols = [str(c) for c in sorted(list(t_vals.intersection(s_vals)))]
    
    if not common_cols:
        raise ValueError("No matching numeric wavelength columns between test file and Embelia standard library.")
        
    is_auth_mask = df_train.iloc[:, 0].str.contains('ribes|Rohini|Rauwolfia', case=False)
    X_auth = df_train[is_auth_mask][common_cols].values
    X_adul = df_train[~is_auth_mask][common_cols].values
    X_test = df_test[common_cols].values
    
    results = []
    for i in range(len(df_test)):
        score, is_auth = authenticate_sample(X_test[i], X_auth, X_adul)
        true_label = df_test.iloc[i, 0] if not str(df_test.iloc[i, 0]).replace('.','').isdigit() else None
        
        results.append({
            "sample_name": f"{filename} (Row {i+1})",
            "prediction": "Authentic Embelia" if is_auth else "Adulterant / Unknown",
            "true_label": true_label,
            "adul_status": "AUTHENTIC" if is_auth else "NON-AUTHENTIC",
            "purity": score, # using match score as pseudo-purity
            "confidence": {"Match Score": score / 100.0},
            "raw_spectrum": X_test[i].tolist(),
            "modality": "UV"
        })
        
    return results, "UV"

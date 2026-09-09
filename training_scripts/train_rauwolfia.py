import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

def train_rauwolfia_model(csv_path, output_models_dir, staging_dir=None):
    csv_path = Path(csv_path)
    write_dir = Path(staging_dir) if staging_dir else Path(output_models_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    labels = np.array(df.iloc[:, 0].astype(str).tolist())
    feature_df = df.iloc[:, 1:].copy()

    col_map = {c: float(c) for c in feature_df.columns if c.replace('.','',1).isdigit()}
    wave_cols = list(col_map.keys())
    orig_waves = np.array(list(col_map.values()))
    raw_vals = feature_df[wave_cols].values.astype(float)

    grid_waves = np.arange(200, 1000, 1)
    X_interp = np.array([np.interp(grid_waves, orig_waves, row) for row in raw_vals])

    means = X_interp.mean(axis=1, keepdims=True)
    stds = X_interp.std(axis=1, keepdims=True) + 1e-8
    X_snv = (X_interp - means) / stds

    imputer = SimpleImputer(strategy='mean')
    X_imp = imputer.fit_transform(X_snv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    n_comp = min(X_scaled.shape[0] - 1, 10)
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    model = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
    
    unique_classes, counts = np.unique(labels, return_counts=True)
    if len(unique_classes) > 1 and min(counts) >= 2:
        n_splits = min(3, min(counts))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_pca, labels, cv=cv)
        cv_acc = float(cv_scores.mean() * 100)
    else:
        cv = KFold(n_splits=min(3, len(labels)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_pca, labels, cv=cv)
        cv_acc = float(cv_scores.mean() * 100)

    model.fit(X_pca, labels)
    y_pred = model.predict(X_pca)
    acc = float(accuracy_score(labels, y_pred))
    cm = confusion_matrix(labels, y_pred)
    classes = list(np.unique(labels))

    joblib.dump(model, write_dir / 'svm_uv.pkl')
    joblib.dump(scaler, write_dir / 'scaler_uv.pkl')
    joblib.dump(pca, write_dir / 'pca_uv.pkl')
    joblib.dump(imputer, write_dir / 'imputer_uv.pkl')

    with open(write_dir / 'reference_standard_uv.json', 'w') as f:
        json.dump({str(int(w)): float(val) for w, val in zip(grid_waves, X_interp.mean(axis=0))}, f)

    X_recon = pca.inverse_transform(X_pca)
    t2_scores = np.array([s @ np.eye(X_pca.shape[1]) @ s for s in X_pca])
    q_scores = np.sum((X_scaled - X_recon) ** 2, axis=1)

    t2_thresh = float(np.percentile(t2_scores, 99)) if len(t2_scores) > 0 else 999.0
    q_thresh = float(np.percentile(q_scores, 99)) if len(q_scores) > 0 else 999.0

    with open(write_dir / 'metadata_uv.json', 'w') as f:
        json.dump({
            'feature_columns': [str(int(w)) for w in grid_waves],
            'inv_cov': np.eye(X_pca.shape[1]).tolist(),
            'n_components': int(X_pca.shape[1]),
            't2_threshold': t2_thresh,
            'q_threshold': q_thresh,
            'species_centroids': {cls: X_pca[labels == cls].mean(axis=0).tolist() for cls in classes}
        }, f)

    return {
        'accuracy': acc * 100,
        'cv_accuracy': cv_acc,
        'confusion_matrix': cm,
        'classes': classes,
        'report': classification_report(labels, y_pred, target_names=classes, output_dict=True)
    }

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    data = root / 'uv_module' / 'data' / 'Rauwolfia_UV.csv'
    models = root / 'uv_module' / 'models'
    res = train_rauwolfia_model(data, models)
    print('Rauwolfia Training Success! CV Accuracy:', res['cv_accuracy'])

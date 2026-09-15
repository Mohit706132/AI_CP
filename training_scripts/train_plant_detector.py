from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

DEFAULT_GRID = np.arange(220.0, 801.0, 2.0)

def extract_std_spectrum(df, grid=DEFAULT_GRID):
    col_map = {c: float(c) for c in df.columns if c.replace('.','',1).isdigit()}
    wave_cols = list(col_map.keys())
    orig_waves = np.array(list(col_map.values()))
    raw_vals = df[wave_cols].values.astype(float)
    X_interp = np.array([np.interp(grid, orig_waves, row) for row in raw_vals])
    means = X_interp.mean(axis=1, keepdims=True)
    stds = X_interp.std(axis=1, keepdims=True) + 1e-8
    return (X_interp - means) / stds

def train_plant_detector_model(data_path_or_dir, output_models_dir, staging_dir=None):
    path = Path(data_path_or_dir)
    write_dir = Path(staging_dir) if staging_dir else Path(output_models_dir)
    write_dir.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        data_dir = path.parent
        custom_df = pd.read_csv(path)
    else:
        data_dir = path
        custom_df = None

    e_file = data_dir / 'Embelia_UV_new.csv'
    r_file = data_dir / 'Rauwolfia_UV.csv'
    h_file = data_dir / 'Terminalia_UV_new.csv'

    e_df = pd.read_csv(e_file) if e_file.exists() else pd.read_csv(data_dir / 'Embelia_UV.csv')
    r_df = pd.read_csv(r_file) if r_file.exists() else pd.read_csv(data_dir / 'Rauwolfia_UV.csv')
    h_df = pd.read_csv(h_file) if h_file.exists() else pd.read_csv(data_dir / 'Terminalia.csv')

    X_e = extract_std_spectrum(e_df)
    y_e = np.array(['Embelia'] * len(e_df))

    X_r = extract_std_spectrum(r_df)
    y_r = np.array(['Rauwolfia'] * len(r_df))

    X_h = extract_std_spectrum(h_df)
    y_h = np.array(['Hirda'] * len(h_df))

    X_list = [X_e, X_r, X_h]
    y_list = [y_e, y_r, y_h]

    if custom_df is not None:
        first_col = custom_df.columns[0]
        c_labels = custom_df[first_col].astype(str).tolist()
        mapped_labels = []
        for lbl in c_labels:
            low = lbl.lower()
            if 'embelia' in low or 'ribes' in low or 'tsjeriam' in low:
                mapped_labels.append('Embelia')
            elif 'rauwolfia' in low or 'serpentina' in low:
                mapped_labels.append('Rauwolfia')
            else:
                mapped_labels.append('Hirda')
        X_custom = extract_std_spectrum(custom_df)
        X_list.append(X_custom)
        y_list.append(np.array(mapped_labels))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    counts = pd.Series(y_all).value_counts().tolist()
    min_count = min(counts) if counts else 0
    if min_count >= 2:
        n_splits = min(5, min_count)
        cv_scores = cross_val_score(model, X_all, y_all, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42))
        cv_acc = float(cv_scores.mean() * 100)
    elif len(y_all) >= 2:
        cv_scores = cross_val_score(model, X_all, y_all, cv=KFold(n_splits=min(len(y_all), 3), shuffle=True, random_state=42))
        cv_acc = float(cv_scores.mean() * 100)
    else:
        cv_acc = 100.0

    model.fit(X_all, y_all)
    y_pred = model.predict(X_all)
    acc = float(accuracy_score(y_all, y_pred))
    cm = confusion_matrix(y_all, y_pred)
    classes = sorted(list(np.unique(y_all)))

    artifact = {
        'grid': DEFAULT_GRID.tolist(),
        'model': model,
        'confidence_threshold': 0.60,
        'classes': classes
    }
    joblib.dump(artifact, write_dir / 'plant_detector.pkl')

    return {
        'accuracy': acc * 100,
        'cv_accuracy': cv_acc,
        'confusion_matrix': cm,
        'classes': classes,
        'report': classification_report(y_all, y_pred, target_names=classes, output_dict=True)
    }

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    data = root / 'uv_module' / 'data'
    models = root / 'uv_module' / 'models'
    res = train_plant_detector_model(data, models)
    print('Plant Detector Training Success! CV Accuracy:', res['cv_accuracy'])

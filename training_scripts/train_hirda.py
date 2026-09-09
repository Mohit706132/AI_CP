import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import savgol_filter

class PhytochemicalRatioTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, include_ratios: bool = True, use_2nd_deriv: bool = True):
        self.include_ratios = include_ratios
        self.use_2nd_deriv = use_2nd_deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df_in = X.copy()
            col_map = {}
            for col in df_in.columns:
                try:
                    col_map[col] = float(col)
                except ValueError:
                    pass
            wave_cols = [c for c in df_in.columns if c in col_map]
            X_arr = df_in[wave_cols].values.astype(np.float64)
            wave_nums = np.array([col_map[c] for c in wave_cols])
        else:
            X_arr = np.asarray(X, dtype=np.float64).copy()
            wave_nums = np.arange(220, 220 + X_arr.shape[1] * 2, 2)[:X_arr.shape[1]]

        if not self.include_ratios:
            return X_arr

        eps = 1e-6
        get_col = lambda w: X_arr[:, np.argmin(np.abs(wave_nums - w))]
        a220, a258, a280, a306, a340 = get_col(220.0), get_col(258.0), get_col(280.0), get_col(306.0), get_col(340.0)

        r_258_280 = a258 / (a280 + eps)
        r_306_220 = a306 / (a220 + eps)
        r_280_340 = a280 / (a340 + eps)

        d1 = savgol_filter(X_arr, window_length=9, polyorder=2, deriv=1, axis=1)
        d2 = savgol_filter(X_arr, window_length=15, polyorder=3, deriv=2, axis=1)

        idx_240 = np.argmin(np.abs(wave_nums - 240.0))
        idx_258 = np.argmin(np.abs(wave_nums - 258.0))
        idx_280 = np.argmin(np.abs(wave_nums - 280.0))
        idx_306 = np.argmin(np.abs(wave_nums - 306.0))
        idx_340 = np.argmin(np.abs(wave_nums - 340.0))

        d2_280_258 = d2[:, idx_280] / (d2[:, idx_258] + eps)
        d2_306_258 = d2[:, idx_306] / (d2[:, idx_258] + eps)
        d2_280_340 = d2[:, idx_280] / (d2[:, idx_340] + eps)
        d2_258_240 = d2[:, idx_258] / (d2[:, idx_240] + eps)

        extra_features = [
            r_258_280, r_306_220, r_280_340,
            d2_280_258, d2_306_258, d2_280_340, d2_258_240
        ]
        return np.column_stack([X_arr] + extra_features)

EXPECTED_WAVELENGTHS = np.arange(220, 802, 2)

def train_hirda_model(csv_path, output_models_dir, n_estimators=100, staging_dir=None):
    csv_path = Path(csv_path)
    write_dir = Path(staging_dir) if staging_dir else Path(output_models_dir) / 'Model'
    write_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    labels_raw = df.iloc[:, 0].astype(str).values
    feature_df = df.iloc[:, 1:].copy()

    col_map = {c: float(c) for c in feature_df.columns if c.replace('.','',1).isdigit()}
    wave_cols = list(col_map.keys())
    orig_waves = np.array(list(col_map.values()))
    raw_vals = feature_df[wave_cols].values.astype(float)

    interp_matrix = [np.interp(EXPECTED_WAVELENGTHS, orig_waves, row) for row in raw_vals]
    X_raw = pd.DataFrame(interp_matrix, columns=[str(int(w)) for w in EXPECTED_WAVELENGTHS])

    ratio_trans = PhytochemicalRatioTransformer(include_ratios=True, use_2nd_deriv=True)
    X_feats = ratio_trans.transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feats)

    pca = PCA(n_components=0.98, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels_raw)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    cv_scores = cross_val_score(model, X_pca, y_encoded, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    model.fit(X_pca, y_encoded)

    y_pred = model.predict(X_pca)
    acc = float(accuracy_score(y_encoded, y_pred))
    cm = confusion_matrix(y_encoded, y_pred)
    classes = list(encoder.classes_)

    sys.modules['__main__'].PhytochemicalRatioTransformer = PhytochemicalRatioTransformer
    joblib.dump(scaler, write_dir / 'scaler.pkl')
    joblib.dump(pca, write_dir / 'pca.pkl')
    joblib.dump(model, write_dir / 'best_model.pkl')
    joblib.dump(encoder, write_dir / 'label_encoder.pkl')
    joblib.dump(ratio_trans, write_dir / 'ratio_transformer.pkl')
    
    meta = {
        'sample_count': len(df),
        'cv_accuracy': float(cv_scores.mean() * 100),
        'classes': classes
    }
    joblib.dump(meta, write_dir / 'pipeline_meta.pkl')

    return {
        'accuracy': acc * 100,
        'cv_accuracy': float(cv_scores.mean() * 100),
        'confusion_matrix': cm,
        'classes': classes,
        'report': classification_report(y_encoded, y_pred, target_names=classes, output_dict=True)
    }

if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    data = root / 'uv_module' / 'data' / 'Terminalia_UV_new.csv'
    models = root / 'uv_module' / 'models'
    res = train_hirda_model(data, models)
    print('Hirda Training Success! CV Accuracy:', res['cv_accuracy'])

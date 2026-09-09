from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PhytochemicalRatioTransformer(BaseEstimator, TransformerMixin):
    """Transformer for engineering 1st and 2nd derivative peak curvature ratio features."""

    def __init__(self, include_ratios: bool = True, use_2nd_deriv: bool = True):
        self.include_ratios = include_ratios
        self.use_2nd_deriv = use_2nd_deriv
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> PhytochemicalRatioTransformer:
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        from scipy.signal import savgol_filter

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

        def get_col(target_wave: float) -> np.ndarray:
            idx = np.argmin(np.abs(wave_nums - target_wave))
            return X_arr[:, idx]

        a220 = get_col(220.0)
        a258 = get_col(258.0)
        a280 = get_col(280.0)
        a306 = get_col(306.0)
        a340 = get_col(340.0)

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

        d1_258 = d1[:, idx_258]
        d1_280 = d1[:, idx_280]

        d2_ratio_280_258 = d2[:, idx_280] / (d2[:, idx_258] + eps)
        d2_ratio_306_258 = d2[:, idx_306] / (d2[:, idx_258] + eps)
        d2_ratio_280_340 = d2[:, idx_280] / (d2[:, idx_340] + eps)
        d2_ratio_258_240 = d2[:, idx_258] / (d2[:, idx_240] + eps)

        if self.use_2nd_deriv:
            engineered_feats = np.column_stack([
                r_258_280,
                r_306_220,
                r_280_340,
                d1_258,
                d1_280,
                d2_ratio_280_258,
                d2_ratio_306_258,
                d2_ratio_280_340,
                d2_ratio_258_240,
            ])
        else:
            engineered_feats = np.column_stack([
                r_258_280,
                r_306_220,
                r_280_340,
                d1_258,
                d1_280,
            ])

        return np.hstack([X_arr, engineered_feats])

# Ensure pickle can load it by putting it in __main__
sys.modules['__main__'].PhytochemicalRatioTransformer = PhytochemicalRatioTransformer

def detect_modality(df):
    cols = []
    for c in df.columns:
        try:
            cols.append(float(c))
        except:
            pass
    if not cols:
        return None
    if max(cols) <= 1200:
        return 'UV'
    return 'FTIR'

def apply_snv(X):
    X_val = np.array(X, dtype=float)
    means = np.mean(X_val, axis=1, keepdims=True)
    stds = np.std(X_val, axis=1, keepdims=True)
    return (X_val - means) / (stds + 1e-8)

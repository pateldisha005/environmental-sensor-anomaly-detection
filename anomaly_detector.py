"""
Anomaly detection module.

Three complementary methods are applied and their scores are ensembled:

  1. Z-score threshold  — simple, interpretable, fast.
     Flags any point where |z| > threshold (default 3σ).
     Good at catching sharp point anomalies (spikes, dropouts).

  2. Isolation Forest   — tree-based unsupervised model.
     Isolates anomalies by recursively partitioning the feature space.
     Anomalies are points that require fewer splits to isolate.
     Good at multivariate contextual anomalies (unusual combination of sensors).

  3. Rolling IQR        — non-parametric, robust to outliers.
     At each timestamp, flags points outside [Q1 - k*IQR, Q3 + k*IQR]
     computed over a rolling window. Works well on non-Gaussian distributions.

Ensemble: a point is labelled anomalous if at least 2 of 3 methods flag it.
This reduces false positives from any single method.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = [
    "temperature_residual_z",
    "humidity_residual_z",
    "wind_speed_residual_z",
    "co2_ppm_residual_z",
]


def zscore_detector(df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
    """Flags points where any sensor's z-score exceeds the threshold."""
    z_cols = [c for c in FEATURE_COLS if c in df.columns]
    max_z  = df[z_cols].abs().max(axis=1)
    return (max_z > threshold).astype(int)


def isolation_forest_detector(df: pd.DataFrame, contamination: float = 0.05) -> pd.Series:
    """
    Isolation Forest trained on z-scored residuals.
    contamination = expected fraction of anomalies (5% is a reasonable default).
    Returns 1 for anomaly, 0 for normal.
    """
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0).values
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    raw = clf.fit_predict(X)   # sklearn uses -1 for anomaly, +1 for normal
    return pd.Series((raw == -1).astype(int), index=df.index)


def rolling_iqr_detector(df: pd.DataFrame, window: int = 24*7, k: float = 2.5) -> pd.Series:
    """
    Non-parametric anomaly detection using rolling interquartile range.
    k=2.5 corresponds roughly to 3σ for a Gaussian but is robust to heavy tails.
    """
    flag_any = pd.Series(0, index=df.index)
    z_cols = [c for c in FEATURE_COLS if c in df.columns]

    for col in z_cols:
        q1  = df[col].rolling(window=window, center=True, min_periods=1).quantile(0.25)
        q3  = df[col].rolling(window=window, center=True, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        flag_any |= ((df[col] < lower) | (df[col] > upper)).astype(int)

    return flag_any


def detect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all three detectors and combines them with majority voting.
    """
    df = df.copy()
    df["flag_zscore"] = zscore_detector(df)
    df["flag_iforest"] = isolation_forest_detector(df)
    df["flag_iqr"] = rolling_iqr_detector(df)

    # Majority vote: flag if ≥ 2 out of 3 methods agree
    vote = df["flag_zscore"] + df["flag_iforest"] + df["flag_iqr"]
    df["predicted_anomaly"] = (vote >= 2).astype(int)

    return df


def evaluate(df: pd.DataFrame):
    """
    Computes precision, recall, F1 if ground-truth 'anomaly' column exists.
    Only meaningful on synthetic data where we injected known anomalies.
    """
    if "anomaly" not in df.columns:
        print("No ground-truth labels — skipping evaluation.")
        return

    from sklearn.metrics import classification_report
    print("=== Anomaly Detection Evaluation ===")
    print(classification_report(df["anomaly"], df["predicted_anomaly"],
                                target_names=["Normal", "Anomaly"]))

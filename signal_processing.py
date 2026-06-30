"""
Signal processing module.

Applies classical DSP techniques to environmental time series before
anomaly detection. This is the bridge between the physics/signal processing
background and the ML layer.

Techniques:
  - FFT spectrum analysis (identify dominant frequencies / periodicities)
  - Low-pass filtering via rolling average (de-noise without edge artefacts)
  - Residual extraction (removes seasonal/trend component to expose anomalies)
  - Z-score standardisation per channel
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

SENSOR_COLS = ["temperature", "humidity", "wind_speed", "co2_ppm"]


def compute_fft(series: pd.Series, sample_rate_hz: float = 1/3600) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the single-sided amplitude spectrum of a uniformly sampled series.

    sample_rate_hz: samples per second. Default = 1 sample/hour.
    Returns (frequencies_in_cycles_per_day, amplitudes)
    """
    values = series.dropna().values
    n = len(values)
    spectrum  = np.abs(np.fft.rfft(values - values.mean())) / n
    freqs_hz  = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    freqs_cpd = freqs_hz * 86400   # convert Hz → cycles/day
    return freqs_cpd, spectrum


def rolling_smooth(df: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """
    Low-pass filter via centred rolling mean.
    Removes sub-daily noise while preserving the trend and seasonal signals.
    """
    df_smooth = df.copy()
    for col in SENSOR_COLS:
        if col in df.columns:
            df_smooth[col + "_smooth"] = (
                df[col].rolling(window=window_hours, center=True, min_periods=1).mean()
            )
    return df_smooth


def compute_residuals(df: pd.DataFrame, window_hours: int = 24 * 7) -> pd.DataFrame:
    """
    Residual = raw signal − long-term rolling mean (trend + seasonal baseline).

    Anomalies are much more visible in the residual than in the raw signal
    because the seasonal variation (±15°C over a year) dwarfs most anomalies (±5°C).
    """
    df_out = df.copy()
    for col in SENSOR_COLS:
        if col in df.columns:
            baseline = df[col].rolling(window=window_hours, center=True, min_periods=1).mean()
            df_out[col + "_residual"] = df[col] - baseline
    return df_out


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises residual columns to zero mean and unit variance.
    Allows multi-sensor anomaly scores to be combined on the same scale.
    """
    df_out = df.copy()
    residual_cols = [c for c in df.columns if c.endswith("_residual")]
    for col in residual_cols:
        mu  = df[col].mean()
        std = df[col].std()
        df_out[col + "_z"] = (df[col] - mu) / (std + 1e-8)
    return df_out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing chain: smooth → residuals → z-score.
    Returns an enriched dataframe ready for the anomaly detector.
    """
    df = rolling_smooth(df)
    df = compute_residuals(df)
    df = zscore(df)
    return df

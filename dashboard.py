"""
Visualization dashboard.

Produces a multi-panel figure showing:
  1. Raw sensor time series with anomalies highlighted
  2. FFT power spectrum (dominant periodicities)
  3. Z-scored residuals with detection threshold
  4. Detector agreement heatmap (which methods flagged which events)
  5. Anomaly density over time (rolling 7-day count)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from signal_processing import compute_fft

OUTPUT_DIR = "outputs"


def plot_sensor_with_anomalies(df: pd.DataFrame, sensor="temperature",
                                save_path=None):
    """Plots raw sensor signal with detected anomaly windows shaded."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["timestamp"], df[sensor], color="#1f77b4", lw=0.8, label=sensor)

    anomaly_mask = df["predicted_anomaly"].values
    in_anomaly = False
    start = None
    for i, flag in enumerate(anomaly_mask):
        if flag and not in_anomaly:
            start = df["timestamp"].iloc[i]
            in_anomaly = True
        elif not flag and in_anomaly:
            ax.axvspan(start, df["timestamp"].iloc[i], alpha=0.3, color="red")
            in_anomaly = False
    if in_anomaly:
        ax.axvspan(start, df["timestamp"].iloc[-1], alpha=0.3, color="red",
                   label="Detected anomaly")

    ax.set_title(f"{sensor.capitalize()} — Detected Anomalies (red shading)")
    ax.set_xlabel("Date")
    ax.set_ylabel(sensor)
    ax.legend()
    plt.tight_layout()
    path = save_path or f"{OUTPUT_DIR}/{sensor}_anomalies.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_fft_spectrum(df: pd.DataFrame, sensor="temperature",
                      save_path=None):
    """
    FFT power spectrum. Expected peaks:
      - 1 cycle/day   — diurnal temperature cycle
      - 7 cycles/week — weekly patterns (CO2 from traffic)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    freqs, amps = compute_fft(df[sensor])

    # Keep only 0–10 cycles/day range (sub-daily to weekly)
    mask = (freqs > 0) & (freqs <= 10)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[mask], amps[mask], color="#2ca02c", lw=0.9)
    ax.axvline(1.0, color="red", ls="--", alpha=0.7, label="1 cycle/day (diurnal)")
    ax.axvline(1/7, color="orange", ls="--", alpha=0.7, label="1 cycle/week (seasonal)")
    ax.set_xlabel("Frequency (cycles per day)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"FFT Spectrum — {sensor.capitalize()}")
    ax.legend()
    plt.tight_layout()
    path = save_path or f"{OUTPUT_DIR}/{sensor}_fft.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_detector_agreement(df: pd.DataFrame, save_path=None):
    """Heatmap showing which detectors fired at each anomalous timestep."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    anom = df[df["predicted_anomaly"] == 1].copy()
    if anom.empty:
        print("No anomalies detected — skipping detector agreement plot.")
        return

    anom = anom[["flag_zscore", "flag_iforest", "flag_iqr"]].head(200)
    anom.columns = ["Z-score", "Isolation Forest", "Rolling IQR"]

    fig, ax = plt.subplots(figsize=(10, max(3, len(anom) // 20)))
    sns.heatmap(anom.T, cbar=False, cmap="Reds", ax=ax, xticklabels=False)
    ax.set_title("Detector Agreement on Flagged Windows")
    ax.set_xlabel("Anomalous timesteps")
    plt.tight_layout()
    path = save_path or f"{OUTPUT_DIR}/detector_agreement.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_anomaly_density(df: pd.DataFrame, save_path=None):
    """Rolling 7-day count of detected anomalies — shows anomaly clustering."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = df.copy()
    df = df.set_index("timestamp")
    density = df["predicted_anomaly"].rolling("7D").sum()

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.fill_between(density.index, density.values, color="salmon", alpha=0.7)
    ax.set_title("Rolling 7-Day Anomaly Count")
    ax.set_xlabel("Date")
    ax.set_ylabel("Anomalies / 7 days")
    plt.tight_layout()
    path = save_path or f"{OUTPUT_DIR}/anomaly_density.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def generate_all(df: pd.DataFrame):
    for sensor in ["temperature", "humidity", "co2_ppm"]:
        plot_sensor_with_anomalies(df, sensor=sensor)
        plot_fft_spectrum(df, sensor=sensor)
    plot_detector_agreement(df)
    plot_anomaly_density(df)

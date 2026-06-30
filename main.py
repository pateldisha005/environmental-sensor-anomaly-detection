"""
Entry point — full pipeline:
  1. Load or generate sensor data
  2. Signal processing (FFT, smoothing, residuals, z-scores)
  3. Anomaly detection (Z-score, Isolation Forest, Rolling IQR ensemble)
  4. Evaluate against ground truth (synthetic data only)
  5. Generate all dashboard plots
"""

from load_data import load
from signal_processing import engineer_features
from anomaly_detector import detect, evaluate
from dashboard import generate_all


def main():
    print("=" * 55)
    print("  Environmental Sensor Anomaly Detector")
    print("=" * 55)

    print("\n[1/4] Loading data...")
    df = load(use_real=True)
    print(f"  {len(df):,} hourly records  |  columns: {list(df.columns)}")

    print("\n[2/4] Signal processing (FFT, residuals, z-scores)...")
    df = engineer_features(df)

    print("\n[3/4] Running anomaly detectors...")
    df = detect(df)
    n_anom = df["predicted_anomaly"].sum()
    pct    = 100 * n_anom / len(df)
    print(f"  Detected {n_anom:,} anomalous hours ({pct:.1f}% of records)")
    print(f"  Z-score: {df['flag_zscore'].sum()}  |  "
          f"IForest: {df['flag_iforest'].sum()}  |  "
          f"IQR: {df['flag_iqr'].sum()}")

    print("\n[4/4] Evaluation + visualization...")
    evaluate(df)
    generate_all(df)

    print("\nDone! All plots saved to outputs/")
    print("Key files:")
    print("  outputs/temperature_anomalies.png  — time series with anomaly windows")
    print("  outputs/temperature_fft.png        — spectral analysis")
    print("  outputs/detector_agreement.png     — which methods agree")
    print("  outputs/anomaly_density.png        — anomaly clustering over time")


if __name__ == "__main__":
    main()

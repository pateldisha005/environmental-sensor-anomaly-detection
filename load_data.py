"""
Data loading module.

Attempts to download real NOAA climate sensor data (Global Summary of the Day).
Falls back to a realistic synthetic dataset if the download fails or is skipped.

Real dataset: NOAA GSOD (Global Surface Summary of Day)
  Hourly-to-daily measurements from thousands of weather stations worldwide.
  Public domain, no API key required.
  https://www.ncei.noaa.gov/products/land-based-station/global-summary-of-the-day
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = "data/sensor_data.csv"


def download_noaa_gsod(station_id="726300-14733", year=2023) -> pd.DataFrame | None:
    """
    Downloads Global Summary of the Day data for a NOAA weather station.
    Station 726300-14733 = Chicago O'Hare International Airport.
    """
    import requests
    url = (
        f"https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv"
    )
    try:
        print(f"Downloading NOAA GSOD data from {url} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), parse_dates=["DATE"])
        df = df.rename(columns={"DATE": "timestamp", "TEMP": "temperature",
                                 "DEWP": "dewpoint", "WDSP": "wind_speed",
                                 "PRCP": "precipitation"})
        df = df[["timestamp", "temperature", "dewpoint", "wind_speed", "precipitation"]].copy()
        df = df.replace(9999.9, np.nan).replace(999.9, np.nan)
        print(f"Downloaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"Download failed ({e}). Using synthetic data instead.")
        return None


def generate_synthetic_data(n_days=365) -> pd.DataFrame:
    """
    Generates a realistic year-long environmental sensor time series.

    Includes:
      - Seasonal temperature variation (sinusoidal)
      - Daily temperature cycle
      - Correlated humidity
      - Wind speed with realistic autocorrelation
      - Injected point anomalies (sudden spikes)
      - Injected contextual anomalies (anomalous for the season)

    Anomaly labels are stored in the 'anomaly' column (0=normal, 1=anomaly).
    In a real deployment, anomalies would not be labelled — the detector finds them.
    Here we store them so we can evaluate the detector's performance.
    """
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    n = len(timestamps)

    day_of_year = np.arange(n) / 24
    hour_of_day = np.arange(n) % 24

    # Temperature: seasonal + diurnal cycle + noise
    seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 90) / 365)   # peak in summer
    diurnal  = 5  * np.sin(2 * np.pi * (hour_of_day - 6)  / 24)    # peak mid-afternoon
    temp = 10 + seasonal + diurnal + rng.normal(0, 1.5, n)

    # Humidity: inversely correlated with temperature
    humidity = 65 - 0.8 * (temp - temp.mean()) + rng.normal(0, 5, n)
    humidity = np.clip(humidity, 0, 100)

    # Wind speed: autocorrelated (weather fronts persist over hours)
    wind = np.zeros(n)
    wind[0] = 8.0
    for i in range(1, n):
        wind[i] = 0.92 * wind[i-1] + rng.exponential(0.8)
    wind = np.abs(wind)

    # CO2 sensor (ppm): diurnal cycle from vegetation/traffic
    co2_base = 415 + 5 * np.sin(2 * np.pi * (hour_of_day - 8) / 24) + rng.normal(0, 2, n)

    anomaly_mask = np.zeros(n, dtype=int)

    # Point anomalies — sudden spikes
    spike_indices = rng.choice(n, size=30, replace=False)
    temp[spike_indices]     += rng.choice([-20, 20], size=30)
    humidity[spike_indices] += rng.choice([-40, 40], size=30)
    anomaly_mask[spike_indices] = 1

    # Contextual anomaly — hot temperatures in winter (day 10-15 of January)
    winter_anomaly = slice(10*24, 15*24)
    temp[winter_anomaly] += 25
    anomaly_mask[winter_anomaly] = 1

    # Sensor dropout — flat-line (common in real sensor networks)
    dropout = slice(200*24, 202*24)
    temp[dropout] = temp[200*24]
    anomaly_mask[dropout] = 1

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "temperature": np.round(temp, 2),
        "humidity":    np.round(humidity, 2),
        "wind_speed":  np.round(wind, 2),
        "co2_ppm":     np.round(co2_base, 1),
        "anomaly":     anomaly_mask,
    })
    return df


def load(use_real=True) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"Loading cached data from {DATA_PATH}")
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

    df = download_noaa_gsod() if use_real else None

    if df is None:
        print("Generating synthetic environmental sensor data...")
        df = generate_synthetic_data()

    df.to_csv(DATA_PATH, index=False)
    print(f"Data saved to {DATA_PATH}")
    return df


if __name__ == "__main__":
    df = load()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df.describe())

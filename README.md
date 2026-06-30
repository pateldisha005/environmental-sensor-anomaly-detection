# Environmental Sensor Anomaly Detector

Detects anomalies in environmental time-series sensor data (temperature, humidity, CO₂, wind) using an ensemble of classical signal processing and machine learning methods.

Built to demonstrate sensor data processing skills applicable to IoT sensor networks, environmental monitoring systems, and physiological signal analysis (FluidAI, Oncoustics).

## Pipeline

```
Raw sensor data (hourly)
    ↓
[FFT analysis]           — identify dominant periodicities (diurnal, seasonal)
    ↓
[Rolling smoothing]      — low-pass filter to separate trend from noise
    ↓
[Residual extraction]    — remove seasonal baseline to expose true deviations
    ↓
[Z-score normalisation]  — put all sensors on the same scale
    ↓
[3-method ensemble]      — Z-score threshold + Isolation Forest + Rolling IQR
    ↓
Anomaly map + dashboard
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

The pipeline auto-downloads real NOAA weather station data; if offline, a synthetic dataset with injected anomalies is used instead.

## Data sources

**Real data (auto-downloaded):** NOAA Global Summary of the Day (GSOD)
- Chicago O'Hare station (2023) — public domain, no API key
- `load_data.py` handles download and caching

**Synthetic fallback:** 365-day, hourly time series with injected anomalies:
- Point spikes (sudden sensor errors)
- Contextual anomalies (e.g., January heatwave)
- Sensor dropouts (flat-line segments)

## Detection methods

| Method | Strength | Weakness |
|--------|----------|----------|
| Z-score (3σ) | Fast, interpretable | Assumes Gaussian residuals |
| Isolation Forest | Multivariate, no distribution assumption | Black box |
| Rolling IQR | Robust to non-Gaussian, adapts to local variance | Needs enough context window |

**Ensemble:** a point is flagged anomalous if ≥ 2 of 3 methods agree. This reduces false positives from any single method.

## Project structure

```
load_data.py           — NOAA download + synthetic data generation
signal_processing.py   — FFT, smoothing, residuals, z-scores
anomaly_detector.py    — Z-score, Isolation Forest, Rolling IQR, ensemble
dashboard.py           — all plots and visualizations
main.py                — end-to-end pipeline
data/                  — cached sensor data (git-ignored)
outputs/               — generated plots (git-ignored)
```

## Key technical decisions

**Why residual-based detection?** Raw sensor values contain a strong seasonal signal (±15°C over a year). An anomaly of +5°C in summer is trivial; the same deviation in January is extreme. Subtracting the rolling baseline makes the same threshold meaningful across seasons.

**Why ensemble voting?** Each method has failure modes. Z-score misses multivariate anomalies; Isolation Forest can struggle with point spikes; IQR is slow to adapt. Majority voting keeps true anomalies (caught by multiple methods) and discards method-specific noise.

**Why FFT?** FFT reveals the dominant frequencies in the signal (1 cycle/day = diurnal cycle, 1 cycle/7 days = weekly patterns). This validates that the data has expected physics before deploying any detector — a critical sanity check for real sensor pipelines.

## Sample outputs

- `outputs/temperature_anomalies.png` — time series with detected anomaly windows
- `outputs/temperature_fft.png` — power spectrum showing diurnal and seasonal peaks
- `outputs/detector_agreement.png` — which of the three methods fired at each anomaly
- `outputs/anomaly_density.png` — rolling 7-day anomaly count (shows clustering)

## Dependencies

`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`

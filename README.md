# 🌍 PEARLS Air Quality Index (AQI) Predictor & MLOps System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks Feature Store](https://img.shields.io/badge/Hopsworks-v3.x-teal.svg?logo=google-cloud&logoColor=white)](https://www.hopsworks.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=white)](https://reactjs.org/)

An enterprise-grade, multi-horizon atmospheric ML forecasting system engineered to deliver 24-hour (+24h), 48-hour (+48h), and 72-hour (+72h) predictions for Fine Particulate Matter ($\text{PM}_{2.5}$) and European AQI. Built on authentic hourly telemetry from Open-Meteo API, Hopsworks Feature Store, and direct Multi-Horizon Model Tournaments.

---

## 🏛️ System Architecture & MLOps Pipeline

The PEARLS AQI system implements an end-to-end MLOps pipeline featuring automated ingestion, feature engineering, online feature store synchronization, automated model tournaments, champion promotion gates, and a real-time REST API serving an executive React command center.

```
                                   ┌───────────────────────────────────┐
                                   │ Open-Meteo Historical & Live API   │
                                   └─────────────────┬─────────────────┘
                                                     │ Hourly Observations
                                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Operational Data Pipeline (src/fetch_raw_data.py & src/feature_pipeline.py)                           │
│  - Ingests 14-day rolling or 730-day historical window (17,520+ hourly rows)                         │
│  - Engineers 42 features: Cyclical time signals, physical dispersion indices, autoregressive lags  │
└────────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Hopsworks Feature Store   │
                               │ (aqi_hourly_features v2)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Multi-Horizon Tournament  │
                               │ LightGBM / XGB / RF / Ridge│
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Hopsworks Model Registry  │
                               │ Champion Bundle Artifact  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Production REST API & Executive Command Center (src/api.py & client/)                                │
│  - Fast, direct inference serving from registered champion bundle                                    │
│  - Dynamic SHAP explainability, mathematical lift benchmarks, and live telemetry badges              │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Model Performance & Tournament Benchmarks

The system is trained on **2 Full Years (730 Days / ~17,520 Hourly Records)** of continuous Open-Meteo atmospheric observations using a strict chronological 70% Train / 15% Validation / 15% Test split to prevent data leakage.

### Holdout Performance Matrix

| Forecast Horizon | Winning Champion Model | Holdout RMSE ($\mu g/m^3$) | Holdout MAE ($\mu g/m^3$) | Holdout $R^2$ Score | Naive Persistence RMSE | Persistence Lift (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Day 1 (+24h)** | **LightGBM Regressor** | **11.07** | **8.36** | **0.4223** | 12.13 | **+8.76%** |
| **Day 2 (+48h)** | **XGBoost Regressor** | **13.26** | **10.28** | **0.1771** | 14.59 | **+9.08%** |
| **Day 3 (+72h)** | **Random Forest** | **13.74** | **10.80** | **0.1355** | 16.45 | **+16.51%** |
| **Overall 72h** | **Multi-Horizon Ensemble** | **12.69** | **9.81** | **0.2450** | 14.39 | **+11.82%** |

---

## 🔬 Feature Engineering Strategy

The system generates 42 domain-specific atmospheric and temporal features:

1. **Autoregressive Lags & Rolling Window Statistics**:
   - $\text{PM}_{2.5}$ Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h.
   - Rolling Aggregations: 6-hour, 12-hour, 24-hour, and 48-hour Rolling Means, Standard Deviations, Minima, Maxima, and Exponential Moving Averages (EMA).
2. **Cyclical Temporal Signal Encoding**:
   - Hour of Day ($\sin, \cos$) transformation: $\sin(2\pi \cdot \text{hour}/24), \cos(2\pi \cdot \text{hour}/24)$
   - Day of Year ($\sin, \cos$) transformation: $\sin(2\pi \cdot \text{doy}/365), \cos(2\pi \cdot \text{doy}/365)$
3. **Physical Atmospheric Dispersion & Trap Indices**:
   - Thermal Inversion Risk Proxy: Atmospheric Surface Pressure ($\text{hPa}$) vs Relative Humidity ($\%$) interaction.
   - Aerosol Trap Indicator: Wind Speed ($10\text{m}$) dispersion index.
4. **Target Stabilization**:
   - Models are trained using $\log(1 + y)$ target transformation ($\text{TransformedTargetRegressor}$) to balance variance in high-concentration particulate spikes.

---

## 🛠️ Technology Stack

- **Machine Learning**: LightGBM, XGBoost, Scikit-Learn (Random Forest, Ridge Regression, TransformedTargetRegressor), SHAP (TreeExplainer)
- **Feature Store & Model Registry**: Hopsworks (Python SDK `hopsworks`)
- **Backend API**: FastAPI, Uvicorn, Pydantic v2, Python ThreadPoolExecutor
- **Frontend Command Center**: React 18, Vite, Tailwind CSS, Lucide Icons, Recharts, Framer Motion
- **Automation / CI/CD**: GitHub Actions (Cron workflows for hourly feature updates and daily automated retraining)

---

## 🚀 Installation & Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Hopsworks Account & API Key (Optional for local warm-cache execution mode)

### 2. Environment Configuration

Clone the repository and set up environment variables:

```bash
git clone https://github.com/your-username/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# Create and activate Python virtual environment
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# On Linux/macOS:
source venv/bin/activate

# Install Python backend dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root directory (refer to `.env.example`):

```ini
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=pearls_aqi
LOCATION_LATITUDE=31.5204
LOCATION_LONGITUDE=74.3587
LOCATION_NAME=Lahore
STATION_NAME=Central_Observation_Point
```

---

## 💻 Operational Execution Commands

### 1. Cold-Start 2-Year Historical Backfill
Fetches 730 days (~17,520 hourly rows) of authentic historical air quality and weather data from Open-Meteo archive endpoints:

```bash
python src/backfill.py --days 730
```

### 2. Operational Hourly Feature Pipeline
Fetches the latest rolling 14-day observation window, merges fresh records into `data/raw_aqi.csv`, updates engineered features (`data/features.parquet`), and synchronizes the online Feature Group `aqi_hourly_features` v2 on Hopsworks:

```bash
python src/feature_pipeline.py
```

### 3. Model Tournament Training & Hopsworks Registration
Runs the multi-horizon model tournament across LightGBM, XGBoost, Random Forest, and Ridge. Validates models against temporal holdout targets, saves the champion bundle locally to `aqi_best_model/model.pkl`, and registers the artifact to Hopsworks Model Registry:

```bash
python src/train_model.py
```

### 4. Launch Production REST API Backend
Starts the FastAPI service on port `8000`:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Launch React Command Center Frontend
In a separate terminal window:

```bash
cd client
npm install
npm run dev
```

The frontend dashboard will be available at `http://localhost:5173`.

---

## 📡 API Reference Endpoint Specifications

| HTTP Method | Endpoint Path | Description | Response Model |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/telemetry` | Returns real-time predictions, current weather, SHAP explainability, and persistence metrics. | `TelemetryResponse` |
| `GET` | `/api/eda` | Serves statistical summary, feature correlations, and diurnal 24-hour profiles. | JSON Summary Object |
| `GET` | `/api/tournament` | Returns model tournament evaluation metrics and baseline lift matrix. | JSON Tournament Matrix |
| `POST` | `/api/subscribe` | Registers user email for threshold alert notifications (`threshold`, `frequency`). | JSON Status Response |
| `POST` | `/api/unsubscribe` | Unsubscribes email address from alert network. | JSON Status Response |

---

## ⚙️ Automated CI/CD Pipelines (GitHub Actions)

The repository incorporates automated workflows in `.github/workflows/`:

1. **Hourly Feature Ingestion (`.github/workflows/feature_pipeline.yml`)**:
   - Triggers hourly via cron `0 * * * *`.
   - Ingests fresh Open-Meteo observations and synchronizes feature updates to Hopsworks Feature Store.

2. **Automated Retraining & Promotion Gate (`.github/workflows/training_pipeline.yml`)**:
   - Triggers daily via cron `0 2 * * *` (02:00 UTC) or on manual workflow dispatch.
   - Retrains candidate models on expanding feature data, checks performance promotion gates against active model RMSE, and registers winning champion bundles.

---

## 🛡️ Data Provenance & Security Policy

- **Zero Synthetic Fallbacks**: All predictions and feature vectors originate directly from authentic atmospheric telemetry.
- **Fail-Safe Inference**: Non-blocking `ThreadPoolExecutor` ensures that network delays on remote feature store calls fallback seamlessly to warm local Parquet caches without hanging HTTP API requests.
- **Credential Protection**: All sensitive tokens and API keys are strictly configured via environment variables (`.env`) and never committed to source control.

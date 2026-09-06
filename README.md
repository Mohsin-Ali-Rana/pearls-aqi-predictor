# PEARLS Air Quality Index (AQI) Predictor & MLOps System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks Feature Store](https://img.shields.io/badge/Hopsworks-v3.x-teal.svg?logo=google-cloud&logoColor=white)](https://www.hopsworks.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19 + Vite](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=white)](https://reactjs.org/)

**PEARLS AQI Predictor** is a multi-horizon atmospheric forecasting and MLOps platform engineered to predict Fine Particulate Matter ($\text{PM}_{2.5}$) and European Air Quality Index (AQI) levels across 24-hour (+24h), 48-hour (+48h), and 72-hour (+72h) lead times for the **Wah Cantt & Taxila Industrial Corridor** (33.77°N, 72.75°E).

The platform ingests hourly numerical weather prediction telemetry from Open-Meteo, synchronizes engineered feature sets to the Hopsworks Cloud Feature Store, executes 3-Direct LightGBM multi-horizon model evaluation, and serves real-time predictions with SHAP feature explainability through a FastAPI backend and React 19 command dashboard.

---

## 🌟 Command Center Dashboard

![PEARLS AQI Command Center Overview](Project_Screenshots/Main%20Front%20Landing%20.png)

*Figure 1: PEARLS AQI Command Center Dashboard rendering real-time telemetry, hero AQI gauge, and 3-day forecast splines.*

---

## 🔑 Key Features & System Capabilities

### 1. 🎯 Direct Multi-Horizon AI Engine (+24h, +48h, +72h)
- **Zero Error Compounding**: Implements three independent, dedicated LightGBM regressor models targeting predictions exactly 24 hours, 48 hours, and 72 hours into the future.
- **Direct Target Alignment**: Eliminates the exponential error compounding and model drift inherent in traditional recursive multi-step forecasting frameworks.

### 2. ☁️ Hopsworks Cloud Feature Store Synchronization (v2)
- **Continuous Feature Ingestion**: Hourly ingestion of raw meteorological and criteria pollutant telemetry from Open-Meteo NWP feeds.
- **34 Engineered Atmospheric Covariates**: Automatic calculation of temporal lags (t-1 to t-24), rolling statistics (6h, 12h, 24h moving averages), wind vectors, relative humidity interaction terms, and thermal boundary layer inversion proxies.
- **Offline/Online Parity**: Guarantees identical feature transformations during model training and real-time online inference.

### 3. 🔍 Explainable AI (XAI) with SHAP TreeExplainer
- **Global Feature Importance**: Computes game-theoretic SHAP Waterfall rankings to highlight primary atmospheric drivers influencing regional air quality.
- **Local Point-in-Time Attribution**: Provides precise, feature-level contribution metrics ($\mu\text{g/m}^3$) for every forecast horizon (+24h, +48h, +72h).

### 4. 📈 Persistence Lift Benchmark & Auto-Promotion Gating
- **Empirical Baseline Validation**: Evaluates candidate model performance against a compulsory Naive Persistence Baseline ($y_{t+H} = y_t$).
- **Automated Model Registry Gating**: Evaluates Root Mean Squared Error (RMSE) reduction percentages; automatically promotes and registers new champion model artifacts to the Hopsworks Model Registry only when candidate lift metrics exceed active production models.

### 5. 📊 Exploratory Data Analysis (EDA) & Diurnal Profiler
- **Diurnal Pollution Heatmaps**: Visualizes hourly particulate intensity profiles across day-of-week and time-of-day dimensions.
- **Atmospheric Correlation Matrices**: Computes feature correlation heatmaps mapping relationships between temperature, humidity, pressure, wind speed, and $\text{PM}_{2.5}$ concentrations.

### 6. 🏆 Multi-Model Tournament Leaderboard
- **Comparative Estimator Benchmarking**: Runs automated model tournaments comparing LightGBM, XGBoost, Random Forest, ARIMA, and Naive Persistence models across RMSE, MAE, and inference latency dimensions.

### 7. ✉️ Automated Early Warning Email Alert Dispatcher
- **Custom Alert Thresholds**: Allows citizens and administrators to configure custom AQI alert triggers (AQI > 100, 150, 200) and dispatch frequencies (e.g., 6-hour intervals).
- **Automated SMTP Mailer**: Dispatches automated subscription welcome confirmations, health precautions, and high-hazard emergency alert emails when atmospheric conditions degrade.

### 8. 🛡️ High-Availability FastAPI Backend & Resilient Caching
- **Sub-Second Caching**: In-memory response caching (`_TELEMETRY_CACHE`) minimizes redundant computation and network roundtrips.
- **Dual-Tier Fallback Strategy**: Gracefully falls back to local warm parquet feature snapshots (`data/features.parquet`) if Hopsworks API latency or network timeouts occur.
- **Production CORS Whitelist**: Configured with explicit origin whitelisting (`allow_credentials=True`) and catch-all exception handlers to ensure CORS preservation.

### 9. ⚙️ Automated CI/CD Pipelines (GitHub Actions)
- **Automated Ingestion**: Scheduled hourly workflows ingest fresh weather data and push updated feature vectors to Hopsworks.
- **Automated Retraining**: Scheduled daily workflows re-evaluate model performance, execute hyperparameter tuning, and trigger model registry updates.

---

## 🏛️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                         1. DATA INGESTION & FEATURE STORE LAYER                   |
|  +-------------------------------+         +-----------------------------------+  |
|  | Open-Meteo NWP Grid Feed      |         | Hopsworks Online Feature Store v2 |  |
|  | (Raw Weather & Chemical Feed) |         | (aqi_hourly_features Group v1/v2) |  |
|  +---------------+---------------+         +-----------------+-----------------+  |
|                  |                                           |                    |
|                  +--------------------+----------------------+                    |
|                                       |                                           |
+---------------------------------------v-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|                         2. FASTAPI BACKEND & INFERENCE LAYER                      |
|  +-----------------------------------------------------------------------------+  |
|  |  FastAPI REST Server (src/api.py)                                           |  |
|  |  - Endpoints: /api/telemetry, /api/eda, /api/tournament, /api/subscribe       |  |
|  |  - In-Memory Response Cache (_TELEMETRY_CACHE)                              |  |
|  |  - Offline Warm Artifact Fallback (data/features.parquet)                    |  |
|  +------------------------------------+----------------------------------------+  |
|                                       |                                           |
|  +------------------------------------v----------------------------------------+  |
|  |  LightGBM 3-Direct Champion Engines (src/inference.py)                       |  |
|  |  - model_24h.pkl (Day 1) | model_48h.pkl (Day 2) | model_72h.pkl (Day 3)       |  |
|  |  - SHAP TreeExplainer Model Interpretability Engine                         |  |
|  +-----------------------------------------------------------------------------+  |
+---------------------------------------|-------------------------------------------+
                                        |
+---------------------------------------v-------------------------------------------+
|                         3. REACT 19 PRESENTATION LAYER                            |
|  +-----------------------------------------------------------------------------+  |
|  |  React 19 SPA (client/src/App.jsx)                                         |  |
|  |  - Hero AQI Arc Gauge & Atmospheric Covariates Grid                          |  |
|  |  - Recharts 3-Day Trajectory Spline with EPA Reference Bands                |  |
|  |  - Global SHAP Waterfall & Point-in-Time Attribution Table                   |  |
|  |  - Early Warning Email Alert Dispatcher Module                               |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 📸 Core Component Previews

| Feature Component | Interface Preview | Description |
| :--- | :---: | :--- |
| **Hero AQI Arc Gauge** | ![Atmospheric Features](Project_Screenshots/Atmosperic%20Features%20.png) | Half-arc AQI meter with 2x2 atmospheric weather covariates grid. |
| **Forecast Cards & Lift** | ![Forecast Cards](Project_Screenshots/Lift%20and%203%20Day%20Frecast%20Cards.png) | Multi-horizon forecast cards (+24h, +48h, +72h) and persistence lift metrics. |
| **Trajectory Chart** | ![Forecast Spline](Project_Screenshots/3%20Day%20forecast%20graph%20.png) | Recharts continuous forecast spline overlaid with EPA threshold reference bands. |
| **SHAP Explainability** | ![SHAP Explainability](Project_Screenshots/Shap%20.png) | SHAP Waterfall chart revealing feature contribution values for predictions. |
| **Model Tournament** | ![Model Tournament](Project_Screenshots/Model%20tournament%20.png) | Evaluates candidate LightGBM, XGBoost, Random Forest, and baseline models. |
| **Alert Dispatcher** | ![Email Alert](Project_Screenshots/Email%20Subscribe%20.png) | Configurable early-warning email alert subscription form. |

---

## 🛠️ Technology Stack

- **Machine Learning**: LightGBM, Scikit-Learn, SHAP, Joblib
- **Feature Store & Registry**: Hopsworks Cloud Feature Store v2 & Model Registry
- **Backend API**: FastAPI, Uvicorn, Pydantic v2, Asynchronous Caching
- **Frontend SPA**: React 19, Vite, Custom CSS Design System, Recharts, Lucide Icons
- **CI/CD & Automation**: GitHub Actions Automated Ingestion & Retraining Workflows

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Mohsin-Ali-Rana/pearls-aqi-predictor.git
cd pearls-aqi-predictor

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:

```ini
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_PROJECT=pearls_aqi
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PORT=443
LOCATION_LATITUDE=33.77
LOCATION_LONGITUDE=72.75
LOCATION_NAME=Wah_Cantt_Taxila
```

---

## 💻 Pipeline Execution Commands

### Ingest Live Features
```bash
python src/feature_pipeline.py
```

### Execute Training & Model Registration
```bash
python src/train_model.py
```

### Run FastAPI Backend
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Run React Frontend
```bash
cd client
npm install
npm run dev
```

---

## 📡 Deployment URLs

- **Live Frontend (Vercel):** [https://pearls-aqi-predictor-psi.vercel.app](https://pearls-aqi-predictor-psi.vercel.app)
- **Live API Backend (Railway):** [https://pearls-aqi-predictor-production.up.railway.app](https://pearls-aqi-predictor-production.up.railway.app)

---

**Author & Certification:**  
**Mohsin Ali** (Software Engineer)

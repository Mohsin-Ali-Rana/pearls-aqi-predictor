# PEARLS Air Quality Index (AQI) Predictor & MLOps System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Hopsworks Feature Store](https://img.shields.io/badge/Hopsworks-v3.x-teal.svg?logo=google-cloud&logoColor=white)](https://www.hopsworks.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19 + Vite](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=white)](https://reactjs.org/)

**PEARLS AQI Predictor** is a multi-horizon atmospheric forecasting and MLOps platform engineered to predict Fine Particulate Matter ($\text{PM}_{2.5}$) and European Air Quality Index (AQI) levels across 24-hour (+24h), 48-hour (+48h), and 72-hour (+72h) lead times for the **Wah Cantt & Taxila Industrial Corridor**.

The platform ingests hourly numerical weather prediction telemetry from Open-Meteo, synchronizes engineered feature sets to the Hopsworks Feature Store, executes 3-Direct LightGBM multi-horizon model evaluation, and serves real-time predictions with SHAP feature explainability through a FastAPI backend and React 19 command dashboard.

---

## 🌟 Command Center Dashboard

![PEARLS AQI Command Center Overview](Project_Screenshots/Main%20Front%20Landing%20.png)

*Figure 1: PEARLS AQI Command Center Dashboard rendering real-time telemetry, hero AQI gauge, and 3-day forecast splines.*

---

## 🏛️ System Architecture

The multi-tier architecture connects data ingestion, cloud feature store synchronization, direct multi-horizon model inference, model registration, and interactive React client presentation.

```
+-----------------------------------------------------------------------------------+
|                         1. DATA INGESTION & FEATURE STORE LAYER                   |
|  +-------------------------------+         +-----------------------------------+  |
|  | Open-Meteo NWP Grid Feed      |         | Hopsworks Online Feature Store v2 |  |
|  | (Raw Weather & Chemical Feed) |         | (aqi_hourly_features Group v2)    |  |
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

## 📸 Core Component Features

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

- **Live Frontend (Vercel):** [https://pearls-aqi-predictor.vercel.app](https://pearls-aqi-predictor.vercel.app)
- **Live API Backend (Railway):** [https://pearls-aqi-predictor-production.up.railway.app](https://pearls-aqi-predictor-production.up.railway.app)

---

**Author & Certification:**  
**Mohsin Ali** (Software Engineer)

# TECHNICAL PROJECT REPORT

**PEARLS AQI PREDICTOR**  
*Proactive Environmental Air Quality Telemetry & Multi-Horizon AI Forecasting Engine*  
**Software Engineering, System Architecture & MLOps Infrastructure Report**  
_________________________________________________________________________________

### Document Metadata & Control Block

| Metadata Field | Value / Details |
| :--- | :--- |
| **System Name** | PEARLS AQI Predictor (Atmospheric Intelligence Platform) |
| **Target Deployment Zone** | Wah Cantt & Taxila Corridor ($33.77^\circ\text{N}, 72.75^\circ\text{E}$) |
| **Document Type** | Comprehensive Software Engineering & MLOps Project Report |
| **Version & Status** | Version 1.0.0 — Production Release (Validated & Deployed) |
| **Author** | Mohsin Ali (Software Engineer) |
| **Technology Stack** | Python 3.11, FastAPI, LightGBM, Hopsworks Feature Store v2, Open-Meteo API, React 19, Vite, Recharts |
| **Repository URL** | https://github.com/Mohsin-Ali-Rana/pearls-aqi-predictor |
| **Production Backend URL** | https://pearls-aqi-predictor-production.up.railway.app |
| **Production Frontend URL** | https://pearls-aqi-predictor.vercel.app |

---

### Revision & Release History

| Version | Date | Author | Description of Release Scope |
| :--- | :--- | :--- | :--- |
| **v0.1-Alpha** | August 15, 2026 | Mohsin Ali | Initial data ingestion pipeline & single-step baseline LightGBM model. |
| **v0.8-Beta** | August 28, 2026 | Mohsin Ali | Integration of Hopsworks Feature Store v2, 3-Direct Multi-Horizon architecture, and React dashboard. |
| **v1.0-Release**| September 6, 2026| Mohsin Ali | Production deployment on Railway/Vercel, CORS overhaul, asynchronous telemetry sync hold, and full test validation. |

---

## 1. Executive Summary

### 1.1 Context & Project Purpose
Ambient air pollution represents a major public health and environmental challenge across urban and industrial regions in South Asia. In industrial-urban corridors such as **Wah Cantt and Taxila, Pakistan**, high concentrations of fine particulate matter ($PM_{2.5}$ and $PM_{10}$) driven by industrial emissions, vehicular traffic, and seasonal atmospheric inversions lead to severe respiratory and cardiovascular health risks. Effective public health protection requires early-warning atmospheric intelligence rather than reactive, delayed reporting.

The **PEARLS AQI Predictor** is a machine learning operations (MLOps) platform designed to deliver real-time atmospheric telemetry and direct multi-horizon ($+24\text{h}$, $+48\text{h}$, $+72\text{h}$) air quality forecasting. By combining continuous satellite and station data ingestion with gradient-boosted decision tree ensembles and cloud feature store synchronization, PEARLS provides municipal authorities, health administrators, and local citizens with actionable, predictive insights into future air quality states.

### 1.2 Core System Capabilities
1. **Direct Multi-Horizon AI Engine:** Deploys three specialized, independent LightGBM regressor models (`model_24h`, `model_48h`, `model_72h`) to predict $PM_{2.5}$ concentrations shifted exactly 24, 48, and 72 hours into the future, eliminating error accumulation inherent in traditional recursive forecasting.
2. **Automated MLOps Feature Pipeline:** Ingests live atmospheric parameters (temperature, relative humidity, surface pressure, wind vectors, criteria pollutants) via Open-Meteo Numerical Weather Prediction (NWP) feeds and synchronizes engineered features with the **Hopsworks Online Feature Store (v2)**.
3. **Mathematical Persistence Benchmark:** Enforces automated model promotion gating by continuously evaluating AI predictions against a naive persistence baseline ($t+H$), requiring a minimum Root Mean Squared Error (RMSE) reduction for production artifact deployment.
4. **Explainable AI (XAI) Transparency:** Incorporates game-theoretic SHAP (SHapley Additive exPlanations) attribution engines to compute both global feature importance rankings and local point-in-time contribution values for every prediction.
5. **High-Performance Serving Infrastructure:** Utilizes an asynchronous FastAPI backend featuring sub-second response caching (`_TELEMETRY_CACHE`) and dual-tier offline parquet fallback (`data/features.parquet`) to guarantee high availability.
6. **Command Dashboard:** Built on React 19 and Vite with custom CSS design tokens, offering half-arc AQI gauges, continuous 3-day trajectory charts with EPA threshold reference bands, multi-model tournament leaderboards, spatial hotspot monitoring, and automated email alert dispatching.

---

## 2. Domain Analysis & Theoretical Background

### 2.1 Environmental Atmospheric Chemistry & Pollution Indicators
Air quality classification is governed by standard mass concentration thresholds of atmospheric pollutants:
* **Fine Particulate Matter ($PM_{2.5}$):** Microscopic airborne particles with aerodynamic diameters $\le 2.5 \, \mu m$. Due to their small size, $PM_{2.5}$ particles penetrate deep into the pulmonary alveoli and enter the cardiovascular system.
* **Coarse Particulate Matter ($PM_{10}$):** Inhalable dust and industrial particulate matter ($\le 10 \, \mu m$).
* **Gaseous Precursors:** Nitrogen Dioxide ($NO_2$), Sulfur Dioxide ($SO_2$), and Ground-Level Ozone ($O_3$), which undergo complex photochemical reactions under varying ambient temperature and solar radiation.

### 2.2 EPA Air Quality Index (AQI) Calculation Methodology
The Air Quality Index (AQI) is derived from $PM_{2.5}$ concentration $C$ using the United States Environmental Protection Agency (US EPA) piecewise linear breakpoint formula:

$$I = \frac{I_{\text{high}} - I_{\text{low}}}{C_{\text{high}} - C_{\text{low}}} (C - C_{\text{low}}) + I_{\text{low}}$$

Where:
* $I$: The calculated Air Quality Index score.
* $C$: The observed or predicted $PM_{2.5}$ concentration ($\mu g/m^3$).
* $C_{\text{low}}, C_{\text{high}}$: The lower and upper concentration breakpoints for the corresponding EPA category.
* $I_{\text{low}}, I_{\text{high}}$: The corresponding AQI score range (e.g., 0–50 Good, 51–100 Moderate, 101–150 Unhealthy for Sensitive Groups, 151–200 Unhealthy, 201–300 Very Unhealthy, 301–500 Hazardous).

---

## 3. Engineering Objectives & Technical Requirements

### 3.1 The Multi-Step Forecasting Error Accumulation Problem
Classical time-series forecasting frameworks typically apply **Recursive Multi-Step Forecasting**, wherein a single model predicts $\hat{y}_{t+1}$, and this predicted value is recursively appended to the feature vector to predict $\hat{y}_{t+2}$, continuing up to step $t+H$. 

In atmospheric modeling, recursive feeding causes **exponential error propagation**. A slight positive bias in the 1-hour prediction alters rolling moving averages and lag features, causing predictions at 48 and 72 hours to compound variance exponentially, leading to model drift.

### 3.2 System Engineering Objectives
To overcome these structural limitations, PEARLS AQI Predictor was engineered to fulfill core technical requirements:

| Requirement Identifier | Core Objective | Target Engineering Metric / Acceptance Criteria |
| :--- | :--- | :--- |
| **REQ-ENG-01** | Direct Multi-Horizon Architecture | Train 3 separate models targeting $y_{t+24}$, $y_{t+48}$, $y_{t+72}$ directly with zero recursive feedback. |
| **REQ-ENG-02** | Hopsworks Feature Store Sync | Ingest and serve feature vectors from Hopsworks Online Feature Store (`aqi_hourly_features` v2). |
| **REQ-ENG-03** | Persistence Lift Auto-Gate | Achieve RMSE reduction over naive persistence ($t+H$) before validating model artifacts. |
| **REQ-ENG-04** | Low Latency API Response | Serve cached telemetry responses in sub-second time. |
| **REQ-ENG-05** | High Availability & Resilience | Implement dual-tier fallback to local warm parquet snapshots (`features.parquet`) on feature store timeouts. |
| **REQ-ENG-06** | Universal Responsive UI | Guarantee clean rendering across Desktop ($1280\text{px}+$ sidebar) and Mobile ($< 1024\text{px}$ slide drawer). |

---

## 4. System Architecture & Component Blueprint

### 4.1 Multi-Tier System Architecture
PEARLS AQI Predictor employs a three-tier architecture isolating Data Ingestion & Storage, Analytical Inference & API Serving, and Client Presentation.

![Main Command Center Interface](Project_Screenshots/Main%20Front%20Landing%20.png)  
*Figure 4.1 — PEARLS AQI Command Center Overview featuring target location telemetry, hero AQI gauge, and 3-day forecast splines.*

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
|  |  - In-Memory Response Cache (_TELEMETRY_CACHE, TTL 5s)                        |  |
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
|                         3. REACT 19 EXECUTIVE PRESENTATION LAYER                  |
|  +-----------------------------------------------------------------------------+  |
|  |  React 19 SPA (client/src/App.jsx)                                         |  |
|  |  - Executive Header & Async Sync Controller Progress Ticker                 |  |
|  |  - Hero AQI Arc Gauge & Atmospheric Covariates Grid                          |  |
|  |  - Recharts 3-Day Trajectory Spline with EPA Severity Reference Bands        |  |
|  |  - Global SHAP Waterfall & Point-in-Time Attribution Table                   |  |
|  |  - Early Warning Email Alert Dispatcher Module                               |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

### 4.2 Technology Selection Matrix

| Subsystem | Selected Technology | Alternative Evaluated | Selection Rationale & Technical Advantage |
| :--- | :--- | :--- | :--- |
| **ML Algorithm** | **LightGBM Regressor** | XGBoost / Random Forest | Faster training speed, lower memory footprint, and native support for continuous tabular splits. |
| **Feature Store** | **Hopsworks (v2)** | Feast / AWS Feature Store | Open-source feature store with offline/online feature parity and pythonic SDK bindings. |
| **Web Backend** | **FastAPI (Python 3.11)** | Flask / Django | Asynchronous `asyncio` execution, native Pydantic validation, and low overhead. |
| **UI Framework** | **React 19 & Vite** | Next.js / Vue.js | Fast build cycles, fine-grained state management, and zero server-side rendering overhead for SPAs. |
| **Data Viz** | **Recharts** | Chart.js / D3.js | Declarative React SVG integration, high render performance, and seamless styling. |

---

## 5. Feature Engineering & Feature Store Pipeline

### 5.1 Hopsworks Feature Store Integration
The feature group `aqi_hourly_features` (v2) maintains engineered feature fields synchronized with Hopsworks Cloud Feature Store.

![Hopsworks Feature Store View](Project_Screenshots/Hopsworks%20Feature%20View%20.png)  
*Figure 5.1 — Hopsworks Cloud Feature Store Console displaying feature group schema and online feature views.*

| Feature Column Name | Data Type | Engineering Calculation / Transformation Logic |
| :--- | :--- | :--- |
| `time` | `TIMESTAMP` | Primary temporal key (UTC hour resolution). |
| `pm2_5` | `FLOAT64` | Observed target concentration ($PM_{2.5}$ in $\mu g/m^3$). |
| `pm10` | `FLOAT64` | Coarse particulate mass concentration ($\mu g/m^3$). |
| `temperature_2m` | `FLOAT64` | Ambient surface air temperature ($^\circ\text{C}$). |
| `relative_humidity_2m` | `FLOAT64` | Relative humidity percentage ($\%$). |
| `surface_pressure` | `FLOAT64` | Ground-level barometric pressure ($hPa$). |
| `wind_speed_10m` | `FLOAT64` | Surface wind velocity ($km/h$). |
| `wind_direction_10m` | `FLOAT64` | Wind direction angle ($0^\circ - 360^\circ$). |
| `pm2_5_lag_1h` | `FLOAT64` | 1-hour prior $PM_{2.5}$ observation ($y_{t-1}$). |
| `pm2_5_lag_24h` | `FLOAT64` | 24-hour diurnal seasonal lag ($y_{t-24}$). |
| `pm2_5_roll_mean_6h` | `FLOAT64` | Moving average over 6-hour window (smoothing). |
| `pm2_5_roll_std_6h` | `FLOAT64` | Volatility standard deviation over 6-hour window. |
| `hour_sin` / `hour_cos` | `FLOAT64` | Trigonometric harmonic transformation ($\sin(2\pi \cdot \text{hour}/24)$). |
| `boundary_condition` | `STRING` | Derived atmospheric inversion stability indicator. |

---

### 5.2 Dynamic Feature Generation Engine
Feature generation is encapsulated in `src/feature_pipeline.py`. Cyclic diurnal transformations are computed mathematically to prevent artificial boundary discontinuities between hour 23 and hour 0:

$$\text{hour}_{\sin} = \sin\left( \frac{2\pi \cdot \text{hour}}{24} \right), \quad \text{hour}_{\cos} = \cos\left( \frac{2\pi \cdot \text{hour}}{24} \right)$$

---

## 6. Machine Learning Pipeline & MLOps Architecture

### 6.1 3-Direct Model Training Strategy
The training script `src/train_model.py` constructs three dedicated dataset matrices by shifting target vectors forward in time:

1. **$+24\text{h}$ Model Target:** $Y_{24}(t) = y(t + 24)$
2. **$+48\text{h}$ Model Target:** $Y_{48}(t) = y(t + 48)$
3. **$+72\text{h}$ Model Target:** $Y_{72}(t) = y(t + 72)$

```python
# Direct Target Shift Logic in src/train_model.py
df_train['target_24h'] = df_train['pm2_5'].shift(-24)
df_train['target_48h'] = df_train['pm2_5'].shift(-48)
df_train['target_72h'] = df_train['pm2_5'].shift(-72)
```

---

### 6.2 Model Promotion Gate & Hopsworks Registry
Before promoting any candidate model bundle to production (`aqi_best_model/model.pkl`), the training pipeline evaluates candidate RMSE against the active production model in Hopsworks Model Registry:

$$\text{Persistence Lift (\%)} = \left( 1 - \frac{\text{RMSE}_{\text{Candidate}}}{\text{RMSE}_{\text{Persistence}}} \right) \times 100$$

If candidate RMSE improves upon the baseline, the model is registered into the Hopsworks Model Registry and promoted to active status.

![Hopsworks Model Registry](Project_Screenshots/Hopsworks%20Model%20Registry%20.png)  
*Figure 6.1 — Hopsworks Model Registry Console displaying registered champion model artifacts.*

---

### 6.3 Model Interpretability via SHAP (SHapley Additive exPlanations)
To ensure transparency, `src/inference.py` integrates `shap.TreeExplainer`. For any feature vector $x$, the local prediction $\hat{f}(x)$ is decomposed as:

$$\hat{f}(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where $\phi_0$ is the baseline expected value and $\phi_i(x)$ is the marginal SHAP value contribution of feature $i$.

---

## 7. REST API Backend Architecture (FastAPI)

### 7.1 OpenAPI Route Specifications

| Method | Endpoint Path | Parameters | Response Payload Description |
| :--- | :--- | :--- | :--- |
| **GET** | `/api/telemetry` | `force: bool` | Returns live AQI, atmospheric covariates, 3-direct forecasts, SHAP values, and persistence lift. |
| **GET** | `/api/eda` | None | Returns historical diurnal heatmaps, weekly medians, and feature correlation matrices. |
| **GET** | `/api/tournament` | None | Returns model evaluation tournament rankings and champion metrics. |
| **POST**| `/api/subscribe` | Body: `{ email, threshold, frequency }` | Registers subscriber email and dispatches confirmation email. |
| **GET** | `/api/health` | None | Operational health check returning system status and timestamp. |

---

## 8. Frontend Architecture & UI Layout (React 19)

### 8.1 Component Hierarchy & State Architecture
The React application follows a modular hierarchy managed by `client/src/App.jsx`:

```
App.jsx (Root Container & Async State Machine)
 ├── Executive Header & Async Sync Progress Ticker
 ├── MLOps Telemetry Stream Status Bar
 ├── Navigation Bar (Dashboard, EDA, Tournament, Alerts)
 ├── Tab 1: Live Dashboard
 │    ├── HeroAQIGauge.jsx (Half-Arc Gauge & 2x2 Weather Grid)
 │    ├── Persistence Lift Benchmark Banner
 │    ├── Multi-Horizon Forecast Cards (+24h, +48h, +72h)
 │    ├── Forecast Trajectory Spline & EPA Reference Bands Chart
 │    ├── ShapWaterfall.jsx (XAI Feature Attribution)
 │    ├── Spatial Hotspot Stations Grid
 │    └── Alert Dispatcher Subscription Form
 ├── Tab 2: Exploratory Data Analysis (CorrelationHeatmap.jsx)
 └── Tab 3: Model Tournament (TournamentChart.jsx)
```

---

## 9. Comprehensive System Component Demonstration

### 9.1 Executive Command Header & Real-Time Sync Controller
Displays target location details (**Wah Cantt / Taxila Region**), geographical coordinates ($33.77^\circ\text{N}, 72.75^\circ\text{E}$), and real-time synchronization status.

![Main Command Center Interface](Project_Screenshots/Main%20Front%20Landing%20.png)  
*Figure 9.1 — Command Center Header and main dashboard view.*

---

### 9.2 Hero Air Quality Arc Gauge & Atmospheric Covariates Grid
Renders the primary half-arc AQI meter, baseline $PM_{2.5}$ concentration, health advisory text, and ambient weather parameters (Temperature, Humidity, Pressure, Wind Speed & Direction).

![Atmospheric Features & Hero Gauge](Project_Screenshots/Atmosperic%20Features%20.png)  
*Figure 9.2 — Hero Air Quality Arc Gauge alongside Atmospheric Features Grid.*

---

### 9.3 Persistence Lift Benchmark & Multi-Horizon Forecast Cards
Displays calculated error reduction over naive persistence across lead times ($+24\text{h}$, $+48\text{h}$, $+72\text{h}$), alongside target timestamps, predicted AQI values, and severity indicators.

![Persistence Lift & Forecast Cards](Project_Screenshots/Lift%20and%203%20Day%20Frecast%20Cards.png)  
*Figure 9.3 — Persistence Lift Benchmark Card and Multi-Horizon Forecast Cards.*

---

### 9.4 3-Day Continuous Forecast Trajectory & EPA Reference Bands Chart
Interactive Recharts Area Chart displaying historical observations transitioning smoothly into the 72-hour forecast spline, overlaid with EPA severity reference bands.

![3-Day Forecast Graph](Project_Screenshots/3%20Day%20forecast%20graph%20.png)  
*Figure 9.4 — 3-Day Forecast Trajectory Chart with EPA threshold bands.*

---

### 9.5 Exploratory Data Analysis & Diurnal Trend Profiler
Provides historical insights including diurnal hourly $PM_{2.5}$ heatmaps, weekly trend profiles, and feature correlation matrices.

![EDA Diurnal Heatmap](Project_Screenshots/EDA%201%20.png)  
*Figure 9.5 — EDA Module displaying hourly diurnal pollution heatmaps.*

![EDA Correlation & Distribution](Project_Screenshots/EDA%202.png)  
*Figure 9.6 — Feature correlation matrix and atmospheric distributions.*

---

### 9.6 SHAP Global Waterfall & Local Feature Attribution Engine
Exposes model decisions using a game-theoretic SHAP Waterfall chart and point-in-time feature attribution tables.

![SHAP Explainability](Project_Screenshots/Shap%20.png)  
*Figure 9.7 — SHAP Waterfall Chart and Feature Attribution breakdown.*

---

### 9.7 Multi-Model Tournament Leaderboard
Compares candidate models (LightGBM, XGBoost, Random Forest, ARIMA Baseline, Naive Persistence) across RMSE, MAE, and latency metrics to validate champion selection.

![Model Tournament Leaderboard](Project_Screenshots/Model%20tournament%20.png)  
*Figure 9.8 — Multi-Model Tournament Leaderboard UI.*

---

### 9.8 Early Warning Hazardous AQI Alert Dispatcher & Email System
Allows users to register email addresses for automated SMTP notifications when predicted AQI exceeds configured thresholds.

![Email Subscription Form](Project_Screenshots/Email%20Subscribe%20.png)  
*Figure 9.9 — Early Warning Email Alert Subscription Form.*

![Subscription Confirmation Received](Project_Screenshots/Email%20ecevied%20subsciption.png)  
*Figure 9.10 — Email Subscription Welcome Notification Received.*

![Health Precautions Email](Project_Screenshots/Email%20Precautions%20Given%20.png)  
*Figure 9.11 — Automated Health Advisory Email Content.*

![Early Warning Alert Email](Project_Screenshots/Email%20Alert%20.png)  
*Figure 9.12 — Automated High-AQI Hazard Warning Email Dispatch.*

---

## 10. Engineering Challenges & Incident Post-Mortems

### 10.1 Multi-Step Recursive Forecast Divergence
* **Issue:** Appending predicted values back into feature vectors caused error compounding over 48–72 hours.
* **Mitigation:** Implemented a **3-Direct Model Architecture**, training separate models directly on shifted horizon targets ($y_{t+24}, y_{t+48}, y_{t+72}$), eliminating recursive error propagation.

### 10.2 Hopsworks Connection Timeouts & Session Cleanups
* **Issue:** Repeated re-authentication calls during API requests caused network timeouts and SSL certificate cleanup errors (`keyStore.jks` missing).
* **Mitigation:** Implemented a **Singleton Hopsworks Connection Manager** (`_HOPSWORKS_PROJECT_CACHE`) to maintain a single authenticated session, combined with a local parquet cache fallback (`data/features.parquet`).

### 10.3 Cross-Origin Resource Sharing (CORS) Policy Mismatches
* **Issue:** Combining wildcard origins (`allow_origins=["*"]`) with `allow_credentials=True` in FastAPI caused browser preflight (OPTIONS) request failures on Vercel deployment.
* **Mitigation:** Updated `CORSMiddleware` configuration to `allow_credentials=False` for wildcard origins, enabling clean browser cross-origin requests.

### 10.4 Container Cold Start & Missing Local Artifacts
* **Issue:** Fresh cloud container deployments on Railway lacked cached model files, causing `FileNotFoundError`.
* **Mitigation:** Updated `load_champion_model_bundle()` to automatically download the active champion model from Hopsworks Model Registry upon container startup if missing locally.

---

## 11. System Verification & Quality Assurance Audit

### 11.1 System Test Suite Evaluation
Automated test suites were executed using `pytest` to validate core components:
* `test_config.py` — Configuration parameter validation.
* `test_feature_pipeline.py` — Covariate transformation & windowing validation.
* `test_train_model.py` — Dataset splitting & model target alignment validation.
* `test_inference.py` — Prediction pipeline & SHAP contribution output validation.
* `test_api.py` — FastAPI endpoint status and response schema validation.

All automated tests passed successfully, confirming system stability.

---

## 12. Deployment Topology & CI/CD Pipelines

### 12.1 Deployment Infrastructure Architecture
The platform is deployed across decoupled cloud services:
* **Frontend:** Hosted on Vercel Edge Network (React 19 single-page application).
* **Backend:** Hosted on Railway Container Cloud (FastAPI REST service).
* **Feature Store & Registry:** Hosted on Hopsworks Cloud (Feature Store v2 & Model Registry).

![GitHub Actions CI/CD Pipeline](Project_Screenshots/Github%20Actions%20Running%20pipelines%20.png)  
*Figure 12.1 — GitHub Actions Automated Ingestion and Training Pipelines.*

---

## 13. Conclusion & Future Work

### 13.1 Operational Conclusion
The **PEARLS AQI Predictor** platform provides a functional MLOps solution for atmospheric air quality forecasting in the Wah Cantt & Taxila region. By leveraging a 3-Direct LightGBM model strategy, Hopsworks cloud feature store synchronization, FastAPI serving layer, and React 19 dashboard, the system provides transparent, multi-horizon air quality intelligence.

### 13.2 Future Enhancements
1. **Deep Learning Exploration:** Evaluating Temporal Fusion Transformers (TFT) for capturing longer-term seasonal dynamics.
2. **Interactive Map Integration:** Adding Mapbox GL layers for dynamic spatial $PM_{2.5}$ heatmaps.
3. **Multi-Channel Alerts:** Extending automated notifications to SMS (Twilio API) and messaging webhooks.

---

**Report Certification:**  
**Mohsin Ali**  
*Software Engineer*  
September 2026

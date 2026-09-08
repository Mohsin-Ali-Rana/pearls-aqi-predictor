import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn

try:
    import src.inference as inference_mod
    from src.utils import convert_pm25_to_aqi, get_aqi_status
    from src.config import LOCATION_NAME, STATION_NAME, LOCATION_LATITUDE, LOCATION_LONGITUDE
    from src.alerts import dispatch_hazardous_aqi_alerts, send_welcome_email, send_unsubscribe_email
    from src.feature_pipeline import run_feature_pipeline
except ImportError:
    import inference as inference_mod
    from utils import convert_pm25_to_aqi, get_aqi_status
    from config import LOCATION_NAME, STATION_NAME, LOCATION_LATITUDE, LOCATION_LONGITUDE
    from alerts import dispatch_hazardous_aqi_alerts, send_welcome_email, send_unsubscribe_email
    from feature_pipeline import run_feature_pipeline


# --- Stage: REST API Serving Layer ---
# FastAPI application instance setup
app = FastAPI(
    title="PEARLS AQI Predictor API",
    description="MLOps Backend serving dynamic multi-model tournament air quality predictions."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pearls-aqi-predictor-psi.vercel.app",
        "https://pearls-aqi-predictor.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        print("Pre-warming telemetry cache on application startup...")
        payload = compute_telemetry_response(force_reload=False)
        _TELEMETRY_CACHE["payload"] = payload
        _TELEMETRY_CACHE["timestamp"] = time.time()
        print("Telemetry cache pre-warmed successfully.")
    except Exception as e:
        print(f"Startup telemetry pre-warming note: {e}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "PEARLS AQI Predictor API",
        "documentation": "/docs",
        "telemetry": "/api/telemetry"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

_TELEMETRY_CACHE = {
    "payload": None,
    "timestamp": 0.0
}
CACHE_TTL_SECONDS = 120.0






# Pydantic data contracts for request and response validation
class ForecastHorizon(BaseModel):
    horizon: str
    aqi: float
    status: str
    color: str
    pm25: Optional[float] = None
    rmse: Optional[float] = None
    healthAdvisory: Optional[str] = None
    healthDetail: Optional[str] = None
    targetTimestamp: Optional[str] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    windSpeed: Optional[float] = None
    modelName: Optional[str] = None






class TrendPoint(BaseModel):
    time: str
    aqi: float
    pm25: float
    category: Optional[str] = "Moderate"
    categoryColor: Optional[str] = "#F59E0B"
    isForecast: Optional[bool] = False






class HotspotStation(BaseModel):
    id: int
    name: str
    aqi: str
    estimationType: Optional[str] = "Estimated spatial bound derived from regional baseline model"
    color: Optional[str] = "#F1F5F9"
    textColor: Optional[str] = "#0F172A"






class ShapFeature(BaseModel):
    feature: str
    importance: float






class ShapContribution(BaseModel):
    feature: str
    contribution: float
    featureValue: Optional[float] = None






class SubscriptionRequest(BaseModel):
    email: str
    threshold: Optional[int] = 100
    frequency: Optional[str] = "6h"






class SystemMetrics(BaseModel):
    completeness: str
    accuracy: str
    status: str
    registry_verified: Optional[bool] = False






class CurrentWeather(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[str] = None
    boundary_condition: Optional[str] = None
    aerosol_risk: Optional[str] = None
    inversion_risk: Optional[str] = None






class PersistenceLift(BaseModel):
    day1_lift_pct: Optional[float] = None
    day1_rmse: Optional[float] = None
    day2_lift_pct: Optional[float] = None
    day2_rmse: Optional[float] = None
    day3_lift_pct: Optional[float] = None
    day3_rmse: Optional[float] = None
    status: str






class TelemetryResponse(BaseModel):
    city: str
    coordinates: Optional[str] = None
    stationName: str
    currentAQI: float
    aqiStatus: str
    aqiColor: str
    aqiDelta: str
    aqi_delta_pct: Optional[float] = 0.0
    pm25: float
    whoStatus: str
    healthAdvisory: str
    healthDetail: str
    confidenceScore: Optional[int] = None
    modelName: str
    featureStoreStatus: str
    last_updated: Optional[str] = None
    source: Optional[str] = "Open-Meteo Live API"
    mode: Optional[str] = "Live Operational Telemetry"
    fallback_used: Optional[bool] = False
    registry_verified: Optional[bool] = False
    forecasts: List[ForecastHorizon]
    trendHistory: List[TrendPoint]
    hotspots: List[HotspotStation]
    shapExplanations: List[ShapFeature] = []
    localShapContributions: List[ShapContribution] = []
    localShapByHorizon: Optional[dict] = None
    currentWeather: Optional[CurrentWeather] = None
    persistenceLift: Optional[PersistenceLift] = None
    systemMetrics: SystemMetrics






# AQI level ke hisaab se hex color define karne ka function
def get_aqi_color(aqi_val: float) -> str:
    if aqi_val <= 50: return "#10B981"
    elif aqi_val <= 100: return "#F59E0B"
    elif aqi_val <= 150: return "#EA580C"
    elif aqi_val <= 200: return "#EF4444"
    return "#8B5CF6"






# Health recommendations generate karne ka function
def get_health_advisory(aqi_val: float) -> tuple[str, str]:
    if aqi_val <= 50:
        return "Good Air Quality", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi_val <= 100:
        return "Acceptable Air Quality", "Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects."
    return "Unhealthy Air Quality", "Everyone may begin to experience health effects."






# Model predictions ko TelemetryResponse payload mein map karne ka main handler
def compute_telemetry_response(force_reload: bool = False) -> TelemetryResponse:
    try:
        ml_output = inference_mod.run_inference(force_model_reload=force_reload)
    except Exception as err:
        raise HTTPException(status_code=503, detail=f"Upstream live telemetry & inference engine error: {err}")

    tactical = ml_output.get("hourly_tactical", [])
    strategic = ml_output.get("strategic_3_day", {})

    if not tactical or "24h" not in strategic:
        raise HTTPException(status_code=503, detail="Inference engine returned incomplete tactical or strategic forecast.")

    raw_conf = ml_output.get("forecast_confidence")
    dynamic_confidence = int(round(float(raw_conf))) if raw_conf is not None else None
    p_metrics = ml_output.get("pipeline_metrics", {})
    dynamic_completeness = p_metrics.get("completeness") or "N/A"
    dynamic_accuracy = p_metrics.get("model_residual_confidence") or "N/A"

    current_pm25 = float(ml_output.get("current_pm25", 0.0))
    current_aqi = float(ml_output.get("current_aqi_val", round(convert_pm25_to_aqi(current_pm25), 1)))

    advisory_title, advisory_detail = get_health_advisory(current_aqi)

    f_24h = strategic["24h"]
    f_48h = strategic["48h"]
    f_72h = strategic["72h"]

    # Trigger automated hazardous AQI threshold breach alerts to active subscribers
    try:
        dispatch_res = dispatch_hazardous_aqi_alerts(current_aqi, float(f_24h["predicted_aqi"]), advisory_title)
        print(f"[Telemetry Dispatcher] Automated subscriber alert check status: {dispatch_res}")
    except Exception as dispatch_err:
        print(f"[Telemetry Dispatcher] Alert dispatch note: {dispatch_err}")

    fallback_used = bool(ml_output.get("fallback_used", False))
    mode_str = str(ml_output.get("mode", "Live Operational Telemetry"))
    source_str = str(ml_output.get("source", "Open-Meteo Live API"))

    feature_store_status = "Operational | Live Open-Meteo Telemetry" if not fallback_used else "Offline / Local Artifact Mode"

    aqi_delta_str = "► 0.0 vs last observation"
    if len(tactical) >= 2:
        step1_aqi = convert_pm25_to_aqi(float(tactical[0]["predicted_pm2_5"]))
        step2_aqi = convert_pm25_to_aqi(float(tactical[1]["predicted_pm2_5"]))
        d_val = round(step2_aqi - step1_aqi, 1)
        if d_val > 0: aqi_delta_str = f"▲ +{d_val} next hour"
        elif d_val < 0: aqi_delta_str = f"▼ {d_val} next hour"

    who_status_str = "WHO Guideline Met (≤ 15 µg/m³)" if current_pm25 <= 15.0 else f"WHO Limit Breached ({round(current_pm25 / 15.0, 1)}x WHO Annual Threshold)"

    return TelemetryResponse(
        city=str(LOCATION_NAME),
        coordinates=f"{LOCATION_LATITUDE}° N, {LOCATION_LONGITUDE}° E",
        stationName=str(STATION_NAME),
        currentAQI=float(current_aqi),
        aqiStatus=get_aqi_status(current_aqi),
        aqiColor=get_aqi_color(current_aqi),
        aqiDelta=aqi_delta_str,
        aqi_delta_pct=float(ml_output.get("aqi_delta_pct", 0.0)),
        pm25=float(current_pm25),
        whoStatus=who_status_str,
        healthAdvisory=advisory_title,
        healthDetail=advisory_detail,
        confidenceScore=dynamic_confidence,
        modelName=f"{ml_output.get('model_name', 'aqi_pm25_predictor')} v{ml_output.get('model_version', 36)}",
        featureStoreStatus=feature_store_status,
        last_updated=ml_output.get("last_updated"),
        source=source_str,
        mode=mode_str,
        fallback_used=fallback_used,
        registry_verified=bool(ml_output.get("registry_verified", False)),
        forecasts=[
            ForecastHorizon(
                horizon="24H",
                aqi=float(f_24h["predicted_aqi"]),
                status=str(f_24h["status"]),
                color=get_aqi_color(f_24h["predicted_aqi"]),
                pm25=float(f_24h.get("predicted_pm2_5", 0.0)),
                rmse=float(f_24h["rmse"]) if f_24h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_24h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_24h["predicted_aqi"])[1],
                targetTimestamp=f_24h.get("target_timestamp"),
                temperature=f_24h.get("temperature"),
                humidity=f_24h.get("humidity"),
                windSpeed=f_24h.get("wind_speed"),
                modelName=f_24h.get("model_name")
            ),
            ForecastHorizon(
                horizon="48H",
                aqi=float(f_48h["predicted_aqi"]),
                status=str(f_48h["status"]),
                color=get_aqi_color(f_48h["predicted_aqi"]),
                pm25=float(f_48h.get("predicted_pm2_5", 0.0)),
                rmse=float(f_48h["rmse"]) if f_48h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_48h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_48h["predicted_aqi"])[1],
                targetTimestamp=f_48h.get("target_timestamp"),
                temperature=f_48h.get("temperature"),
                humidity=f_48h.get("humidity"),
                windSpeed=f_48h.get("wind_speed"),
                modelName=f_48h.get("model_name")
            ),
            ForecastHorizon(
                horizon="72H",
                aqi=float(f_72h["predicted_aqi"]),
                status=str(f_72h["status"]),
                color=get_aqi_color(f_72h["predicted_aqi"]),
                pm25=float(f_72h.get("predicted_pm2_5", 0.0)),
                rmse=float(f_72h["rmse"]) if f_72h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_72h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_72h["predicted_aqi"])[1],
                targetTimestamp=f_72h.get("target_timestamp"),
                temperature=f_72h.get("temperature"),
                humidity=f_72h.get("humidity"),
                windSpeed=f_72h.get("wind_speed"),
                modelName=f_72h.get("model_name")
            ),
        ],
        trendHistory=[
            TrendPoint(
                time=str(obs["time"]),
                aqi=float(obs["aqi"]),
                pm25=float(obs["pm25"]),
                category=get_aqi_status(float(obs["aqi"])),
                categoryColor=get_aqi_color(float(obs["aqi"])),
                isForecast=False
            ) for obs in ml_output.get("historical_observations", [])[-12:]
        ] + [
            TrendPoint(
                time="Now (Observed)",
                aqi=float(round(current_aqi, 1)),
                pm25=float(round(current_pm25, 1)),
                category=get_aqi_status(current_aqi),
                categoryColor=get_aqi_color(current_aqi),
                isForecast=False
            ),
            TrendPoint(
                time="+24h (Day 1)",
                aqi=float(f_24h["predicted_aqi"]),
                pm25=float(f_24h.get("predicted_pm2_5", 0.0)),
                category=str(f_24h["status"]),
                categoryColor=get_aqi_color(f_24h["predicted_aqi"]),
                isForecast=True
            ),
            TrendPoint(
                time="+48h (Day 2)",
                aqi=float(f_48h["predicted_aqi"]),
                pm25=float(f_48h.get("predicted_pm2_5", 0.0)),
                category=str(f_48h["status"]),
                categoryColor=get_aqi_color(f_48h["predicted_aqi"]),
                isForecast=True
            ),
            TrendPoint(
                time="+72h (Day 3)",
                aqi=float(f_72h["predicted_aqi"]),
                pm25=float(f_72h.get("predicted_pm2_5", 0.0)),
                category=str(f_72h["status"]),
                categoryColor=get_aqi_color(f_72h["predicted_aqi"]),
                isForecast=True
            )
        ],
        hotspots=[HotspotStation(**h) for h in ml_output.get("sensor_hotspots", [])],
        shapExplanations=[
            ShapFeature(feature=item.get("feature", "F"), importance=float(item.get("importance", 0.0)))
            for item in ml_output.get("shap_explanations", [])
        ],
        localShapContributions=[
            ShapContribution(
                feature=item.get("feature", "F"),
                contribution=float(item.get("contribution", 0.0)),
                featureValue=float(item.get("feature_value", 0.0)) if item.get("feature_value") is not None else None
            )
            for item in ml_output.get("local_shap_contributions", [])
        ],
        localShapByHorizon=ml_output.get("local_shap_by_horizon"),
        currentWeather=CurrentWeather(**ml_output["current_weather"]) if "current_weather" in ml_output else None,
        persistenceLift=PersistenceLift(**ml_output["persistence_lift"]) if "persistence_lift" in ml_output else None,
        systemMetrics=SystemMetrics(
            completeness=str(dynamic_completeness),
            accuracy=str(dynamic_accuracy),
            status=f"System Status: {'Operational | Live Open-Meteo Telemetry' if not fallback_used else 'Offline / Local Artifact Mode'}",
            registry_verified=bool(ml_output.get("registry_verified", False))
        )
    )






# React dashboard ke liye main live telemetry API route
@app.get("/api/telemetry", response_model=TelemetryResponse)
def get_live_telemetry(force: bool = False):
    if force or _TELEMETRY_CACHE["payload"] is None or (time.time() - _TELEMETRY_CACHE["timestamp"]) > CACHE_TTL_SECONDS:
        try:
            response = compute_telemetry_response(force_reload=False)
            _TELEMETRY_CACHE["payload"] = response
            _TELEMETRY_CACHE["timestamp"] = time.time()
            return response
        except Exception as e:
            if _TELEMETRY_CACHE["payload"] is not None:
                return _TELEMETRY_CACHE["payload"]
            raise HTTPException(status_code=503, detail=f"Upstream Telemetry Error: {str(e)}")

    return _TELEMETRY_CACHE["payload"]






# Exploratory Data Analysis summary report API route
@app.get("/api/eda")
def get_eda_summary():
    import os, json, time
    eda_json_path = os.path.join("data", "eda_summary.json")
    if os.path.exists(eda_json_path):
        mtime = os.path.getmtime(eda_json_path)
        if (time.time() - mtime) < 86400:
            with open(eda_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

    try:
        from src.eda import run_eda
        return run_eda(save_json=True)
    except Exception as e:
        if os.path.exists(eda_json_path):
            with open(eda_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=503, detail=f"EDA analysis refresh note: {str(e)}")






# Model tournament leaderboard summary API route
@app.get("/api/tournament")
def get_tournament_summary():
    import os, json, time
    t_path = os.path.join("data", "tournament_summary.json")
    if os.path.exists(t_path):
        mtime = os.path.getmtime(t_path)
        with open(t_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    raise HTTPException(status_code=503, detail="Tournament summary pending execution.")






# User email alert subscription API route
@app.post("/api/subscribe")
def subscribe_user_email(req: SubscriptionRequest):
    import os, json
    email = req.email.strip().lower()
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email format. Please enter a valid observer email address (e.g. user@domain.com).")

    try:
        threshold = int(req.threshold) if req.threshold is not None else 100
    except (ValueError, TypeError):
        threshold = 100
    frequency = req.frequency if req.frequency else "6h"

    os.makedirs("data", exist_ok=True)
    sub_file = os.path.join("data", "subscribers.json")
    subscribers = []
    if os.path.exists(sub_file):
        try:
            with open(sub_file, "r") as f:
                subscribers = json.load(f)
        except Exception:
            subscribers = []

    existing_index = -1
    for i, sub in enumerate(subscribers):
        sub_email = (sub if isinstance(sub, str) else sub.get("email", "")).strip().lower()
        if sub_email == email:
            existing_index = i
            break

    is_update = False
    if existing_index >= 0:
        existing_sub = subscribers[existing_index]
        existing_thresh = existing_sub.get("threshold", 100) if isinstance(existing_sub, dict) else 100
        existing_freq = existing_sub.get("frequency", "6h") if isinstance(existing_sub, dict) else "6h"

        if existing_thresh == threshold and existing_freq == frequency:
            welcome_res = send_welcome_email(email, threshold, frequency, is_update=False)
            if welcome_res.get("status") == "error":
                raise HTTPException(status_code=500, detail=f"Email dispatch error: {welcome_res.get('reason')}")
            return {
                "status": "success",
                "message": f"This email ({email}) is already active with these settings.",
                "is_update": False,
                "already_subscribed": True,
                "threshold": threshold,
                "frequency": frequency,
                "email_delivery": welcome_res
            }
        
        last_sent = existing_sub.get("last_sent", 0.0) if isinstance(existing_sub, dict) else 0.0
        subscribers[existing_index] = {"email": email, "threshold": threshold, "frequency": frequency, "last_sent": last_sent}
        is_update = True
    else:
        subscribers.append({"email": email, "threshold": threshold, "frequency": frequency, "last_sent": 0.0})

    with open(sub_file, "w") as f:
        json.dump(subscribers, f, indent=2)

    welcome_res = send_welcome_email(email, threshold, frequency, is_update=is_update)

    if welcome_res.get("status") == "error":
        raise HTTPException(status_code=500, detail=f"Subscribed {email}, but notification email failed to send: {welcome_res.get('reason')}")

    msg_text = f"Successfully updated alert preferences for {email}!" if is_update else f"Successfully subscribed {email}!"
    if welcome_res.get("status") == "dry_run":
        msg_text += " (Note: SMTP credentials unconfigured on backend server; email delivery simulated)."

    return {
        "status": "success",
        "message": msg_text,
        "is_update": is_update,
        "threshold": threshold,
        "frequency": frequency,
        "email_delivery": welcome_res
    }






# User email alert unsubscription API route
@app.post("/api/unsubscribe")
def unsubscribe_user_email(req: SubscriptionRequest):
    import os, json
    email = req.email.strip().lower()
    if not email or "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email format. Please enter a valid registered email address.")

    sub_file = os.path.join("data", "subscribers.json")
    if not os.path.exists(sub_file):
        raise HTTPException(status_code=404, detail="No active subscribers found in system.")

    try:
        with open(sub_file, "r") as f:
            subscribers = json.load(f)
    except Exception:
        subscribers = []

    updated_subscribers = [
        s for s in subscribers
        if (s if isinstance(s, str) else s.get("email", "")).strip().lower() != email
    ]

    if len(updated_subscribers) == len(subscribers):
        raise HTTPException(status_code=404, detail=f"Email '{email}' not found in active subscriber list.")

    with open(sub_file, "w") as f:
        json.dump(updated_subscribers, f, indent=2)

    unsub_res = send_unsubscribe_email(email)

    if unsub_res.get("status") == "error":
        raise HTTPException(status_code=500, detail=f"Unsubscribed {email}, but confirmation email failed to send: {unsub_res.get('reason')}")

    msg_text = f"Successfully unsubscribed {email}."
    if unsub_res.get("status") == "dry_run":
        msg_text += " (Note: SMTP credentials unconfigured on backend server environment)."

    return {
        "status": "success",
        "message": msg_text,
        "email_delivery": unsub_res
    }






if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
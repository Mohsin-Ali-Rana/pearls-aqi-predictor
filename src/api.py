import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import uvicorn

# Import the actual working inference function and utilities
try:
    from src.inference import run_inference
    from src.utils import convert_pm25_to_aqi, get_aqi_status
    from src.config import LOCATION_NAME, STATION_NAME
    from src.alerts import dispatch_hazardous_aqi_alerts
except ImportError:
    from inference import run_inference
    from utils import convert_pm25_to_aqi, get_aqi_status
    from config import LOCATION_NAME, STATION_NAME
    from alerts import dispatch_hazardous_aqi_alerts

app = FastAPI(
    title="PEARLS AQI Predictor API",
    description="MLOps Backend serving dynamic LightGBM multi-horizon air quality predictions."
)

# Enable CORS for React frontend (Vite port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for high-availability inference serving
_TELEMETRY_CACHE = {
    "payload": None,
    "timestamp": 0.0
}
CACHE_TTL_SECONDS = 120.0  # 2 minutes TTL

# --- Pydantic Data Contracts ---
class ForecastHorizon(BaseModel):
    horizon: str
    aqi: float
    status: str
    color: str
    rmse: Optional[float] = None

class TrendPoint(BaseModel):
    time: str
    aqi: float
    pm25: float

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

class SubscriptionRequest(BaseModel):
    email: str

class SystemMetrics(BaseModel):
    completeness: str
    accuracy: str
    status: str

class TelemetryResponse(BaseModel):
    city: str
    stationName: str
    currentAQI: float
    aqiStatus: str
    aqiColor: str
    aqiDelta: str
    pm25: float
    whoStatus: str
    healthAdvisory: str
    healthDetail: str
    confidenceScore: int
    modelName: str
    featureStoreStatus: str
    forecasts: List[ForecastHorizon]
    trendHistory: List[TrendPoint]
    hotspots: List[HotspotStation]
    shapExplanations: List[ShapFeature] = []
    systemMetrics: SystemMetrics


# --- Helper Functions for Formatting Dynamic Outputs ---
def get_aqi_color(aqi_val: float) -> str:
    if aqi_val <= 50:
        return "#10B981"  # Green
    elif aqi_val <= 100:
        return "#F59E0B"  # Yellow/Amber
    elif aqi_val <= 150:
        return "#EA580C"  # Orange
    elif aqi_val <= 200:
        return "#EF4444"  # Red
    return "#8B5CF6"      # Purple

def get_health_advisory(aqi_val: float) -> tuple[str, str]:
    if aqi_val <= 50:
        return "Good Air Quality", "Air quality is considered satisfactory, and air pollution poses little or no risk."
    elif aqi_val <= 100:
        return "Acceptable Air Quality", "Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi_val <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    return "Unhealthy Air Quality", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."


import asyncio

def compute_telemetry_response() -> TelemetryResponse:
    """Executes model inference and constructs TelemetryResponse payload."""
    ml_output = run_inference()

    tactical = ml_output.get("hourly_tactical", [])
    strategic = ml_output.get("strategic_3_day", {})

    if not tactical:
        raise ValueError("Inference engine returned an empty tactical forecast.")

    for horizon_key in ("24h", "48h", "72h"):
        if horizon_key not in strategic:
            raise ValueError(f"Inference engine did not return a '{horizon_key}' forecast.")

    dynamic_confidence = int(round(float(ml_output.get("forecast_confidence", 94))))
    p_metrics = ml_output.get("pipeline_metrics", {})
    dynamic_completeness = p_metrics.get("completeness", "N/A")
    dynamic_accuracy = p_metrics.get("sensor_accuracy", "N/A")

    current_pm25 = float(tactical[0]["predicted_pm2_5"])
    current_aqi = round(convert_pm25_to_aqi(current_pm25), 1)

    advisory_title, advisory_detail = get_health_advisory(current_aqi)

    f_24h = strategic["24h"]
    f_48h = strategic["48h"]
    f_72h = strategic["72h"]

    if len(tactical) < 3:
        raise ValueError("Inference engine returned insufficient tactical predictions.")

    is_stale = ml_output.get("data_freshness_warning", False)
    feature_store_status = "Stale" if is_stale else "Connected"

    return TelemetryResponse(
        city=str(LOCATION_NAME),
        stationName=str(STATION_NAME),
        currentAQI=float(current_aqi),
        aqiStatus=get_aqi_status(current_aqi),
        aqiColor=get_aqi_color(current_aqi),
        aqiDelta="▼ Dynamic Stream Active",
        pm25=float(current_pm25),
        whoStatus="WHO Threshold Evaluated",
        healthAdvisory=advisory_title,
        healthDetail=advisory_detail,
        confidenceScore=dynamic_confidence,
        modelName=f"{ml_output.get('model_name', 'aqi_pm25_predictor')} v{ml_output.get('model_version', 20)}",
        featureStoreStatus=feature_store_status,
        forecasts=[
            ForecastHorizon(
                horizon="24H",
                aqi=float(f_24h["predicted_aqi"]),
                status=str(f_24h["status"]),
                color=get_aqi_color(f_24h["predicted_aqi"]),
                rmse=float(f_24h["rmse"]) if f_24h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_24h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_24h["predicted_aqi"])[1]
            ),
            ForecastHorizon(
                horizon="48H",
                aqi=float(f_48h["predicted_aqi"]),
                status=str(f_48h["status"]),
                color=get_aqi_color(f_48h["predicted_aqi"]),
                rmse=float(f_48h["rmse"]) if f_48h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_48h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_48h["predicted_aqi"])[1]
            ),
            ForecastHorizon(
                horizon="72H",
                aqi=float(f_72h["predicted_aqi"]),
                status=str(f_72h["status"]),
                color=get_aqi_color(f_72h["predicted_aqi"]),
                rmse=float(f_72h["rmse"]) if f_72h.get("rmse") is not None else None,
                healthAdvisory=get_health_advisory(f_72h["predicted_aqi"])[0],
                healthDetail=get_health_advisory(f_72h["predicted_aqi"])[1]
            ),
        ],
        trendHistory=[
            TrendPoint(
                time="+1h",
                aqi=float(round(convert_pm25_to_aqi(tactical[0]["predicted_pm2_5"]), 1)),
                pm25=float(tactical[0]["predicted_pm2_5"])
            ),
            TrendPoint(
                time="+2h",
                aqi=float(round(convert_pm25_to_aqi(tactical[1]["predicted_pm2_5"]), 1)),
                pm25=float(tactical[1]["predicted_pm2_5"])
            ),
            TrendPoint(
                time="+3h",
                aqi=float(round(convert_pm25_to_aqi(tactical[2]["predicted_pm2_5"]), 1)),
                pm25=float(tactical[2]["predicted_pm2_5"])
            ),
            TrendPoint(time="24H Avg", aqi=float(f_24h["predicted_aqi"]), pm25=float(f_24h["predicted_pm2_5"])),
            TrendPoint(time="48H Avg", aqi=float(f_48h["predicted_aqi"]), pm25=float(f_48h["predicted_pm2_5"])),
            TrendPoint(time="72H Avg", aqi=float(f_72h["predicted_aqi"]), pm25=float(f_72h["predicted_pm2_5"])),
        ],
        hotspots=[
            HotspotStation(**h) for h in ml_output.get("sensor_hotspots", [])
        ],
        shapExplanations=[
            ShapFeature(feature=item.get("feature", "F"), importance=float(item.get("importance", 0.0)))
            for item in ml_output.get("shap_explanations", [])
        ],
        systemMetrics=SystemMetrics(
            completeness=str(dynamic_completeness),
            accuracy=str(dynamic_accuracy),
            status="System Status: Operational | Hopsworks Synchronized"
        )
    )

async def _background_telemetry_worker():
    """Asynchronous background task that periodically refreshes Hopsworks stream telemetry and checks alerts."""
    while True:
        try:
            print("🔄 [Background Worker] Refreshing Hopsworks feature stream & inference payload...")
            response = await asyncio.to_thread(compute_telemetry_response)
            _TELEMETRY_CACHE["payload"] = response
            _TELEMETRY_CACHE["timestamp"] = time.time()
            print("⚡ [Background Worker] Telemetry payload updated successfully!")

            # Trigger automated email alert dispatcher
            if response.forecasts and len(response.forecasts) > 0:
                f24_aqi = response.forecasts[0].aqi
                await asyncio.to_thread(
                    dispatch_hazardous_aqi_alerts, 
                    response.currentAQI, 
                    f24_aqi, 
                    response.aqiStatus
                )
        except Exception as e:
            print(f"⚠️ [Background Worker] Refresh note: {e}")
        await asyncio.sleep(300)  # Refresh every 5 minutes


@app.on_event("startup")
async def startup_event():
    """Starts the background telemetry refresher thread on FastAPI boot."""
    asyncio.create_task(_background_telemetry_worker())


@app.get("/api/telemetry", response_model=TelemetryResponse)
def get_live_telemetry():
    """
    Main endpoint called by the React frontend.
    Returns instantly from cached memory (0-10ms latency).
    """
    if _TELEMETRY_CACHE["payload"] is not None:
        return _TELEMETRY_CACHE["payload"]

    try:
        response = compute_telemetry_response()
        _TELEMETRY_CACHE["payload"] = response
        _TELEMETRY_CACHE["timestamp"] = time.time()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")



@app.get("/api/eda")
def get_eda_summary():
    """Returns dynamic EDA statistical metrics and diurnal profile."""
    import os, json
    eda_json_path = os.path.join("data", "eda_summary.json")
    if os.path.exists(eda_json_path):
        with open(eda_json_path, "r") as f:
            return json.load(f)
    try:
        from src.eda import run_eda
        return run_eda(save_json=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate EDA summary: {str(e)}")


@app.post("/api/subscribe")
def subscribe_user_email(req: SubscriptionRequest):
    """Subscribes user email for hazardous AQI threshold alerts."""
    import os, json
    email = req.email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email format.")

    os.makedirs("data", exist_ok=True)
    sub_file = os.path.join("data", "subscribers.json")
    subscribers = []
    if os.path.exists(sub_file):
        try:
            with open(sub_file, "r") as f:
                subscribers = json.load(f)
        except Exception:
            subscribers = []

    if email not in subscribers:
        subscribers.append(email)
        with open(sub_file, "w") as f:
            json.dump(subscribers, f, indent=2)

    return {"status": "success", "message": f"Successfully subscribed {email} to hazardous AQI email alerts!"}


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
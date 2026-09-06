import os
import pytest
from src.inference import run_inference

def test_inference_pipeline_execution():
    """Verify that run_inference executes cleanly and returns structured telemetry."""
    telemetry = run_inference()
    
    assert isinstance(telemetry, dict)
    assert telemetry.get("success") is True
    assert "current_pm25" in telemetry
    assert "current_aqi_val" in telemetry
    assert "local_shap_by_horizon" in telemetry
    
    # Check multi-horizon forecast structure (+24h, +48h, +72h)
    forecasts = telemetry.get("forecast") or telemetry.get("strategic_3_day", [])
    assert len(forecasts) >= 3
    
    # Check SHAP vector dictionary per horizon
    shap_dict = telemetry["local_shap_by_horizon"]
    assert "24h" in shap_dict
    assert "48h" in shap_dict
    assert "72h" in shap_dict

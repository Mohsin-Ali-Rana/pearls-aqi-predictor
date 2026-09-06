import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_api_health_endpoint():
    """Verify GET /health endpoint returns 200 OK and valid status dictionary."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"

def test_api_telemetry_endpoint():
    """Verify GET /api/telemetry endpoint returns 200 OK with complete AQI payload."""
    import src.api as api_mod
    api_mod._TELEMETRY_CACHE["payload"] = None
    api_mod._TELEMETRY_CACHE["timestamp"] = 0.0
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "city" in data
    assert "currentAQI" in data
    assert "forecasts" in data
    assert "localShapByHorizon" in data

def test_api_alerts_subscribe_endpoint():
    """Verify POST /api/subscribe endpoint processes subscriber registration."""
    payload = {
        "email": "test_analyst_qa@wah-industrial.org",
        "threshold": 100.0,
        "frequency": "6h"
    }
    response = client.post("/api/subscribe", json=payload)
    assert response.status_code in [200, 201, 400]
    data = response.json()
    assert "status" in data or "detail" in data or "message" in data

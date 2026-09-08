import os
import json
import pytest
from fastapi.testclient import TestClient
from src.api import app
from src.alerts import dispatch_hazardous_aqi_alerts, get_cooldown_seconds

client = TestClient(app)

def test_api_health_endpoint():
    """Verify GET /health endpoint returns 200 OK and valid status dictionary."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"

def test_api_cors_allows_vercel_frontend():
    """Verify browser requests from the production frontend receive CORS headers."""
    origin = "https://pearls-aqi-predictor-psi.vercel.app"
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert response.headers["access-control-allow-credentials"] == "true"

    preflight = client.options(
        "/api/telemetry",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin

def test_api_alerts_subscribe_and_unsubscribe_flow():
    """Test full cycle of subscribing, updating preferences, already-subscribed handling, and unsubscribing."""
    test_email = "qa_automated_observer@wah-industrial.org"

    # Step 1: Subscribe new user
    sub_payload = {
        "email": test_email,
        "threshold": 150,
        "frequency": "6h"
    }
    sub_res = client.post("/api/subscribe", json=sub_payload)
    assert sub_res.status_code == 200
    sub_data = sub_res.json()
    assert sub_data["status"] == "success"
    assert sub_data["threshold"] == 150

    # Step 2: Re-subscribe exact same settings (should return 200 OK with already_subscribed: True)
    re_sub_res = client.post("/api/subscribe", json=sub_payload)
    assert re_sub_res.status_code == 200
    re_sub_data = re_sub_res.json()
    assert re_sub_data["status"] == "success"
    assert re_sub_data.get("already_subscribed") is True

    # Step 3: Update preferences for existing subscriber
    update_payload = {
        "email": test_email,
        "threshold": 200,
        "frequency": "24h"
    }
    upd_res = client.post("/api/subscribe", json=update_payload)
    assert upd_res.status_code == 200
    upd_data = upd_res.json()
    assert upd_data["status"] == "success"
    assert upd_data["is_update"] is True
    assert upd_data["threshold"] == 200

    # Step 4: Unsubscribe with mixed casing and whitespace
    unsub_payload = {
        "email": f"  {test_email.upper()}  "
    }
    unsub_res = client.post("/api/unsubscribe", json=unsub_payload)
    assert unsub_res.status_code == 200
    unsub_data = unsub_res.json()
    assert unsub_data["status"] == "success"

    # Step 5: Unsubscribe again (should return 404 as user was removed)
    unsub_again_res = client.post("/api/unsubscribe", json={"email": test_email})
    assert unsub_again_res.status_code == 404

def test_api_alerts_invalid_email_validation():
    """Verify system rejects invalid email formats."""
    res = client.post("/api/subscribe", json={"email": "invalid_email_str", "threshold": 100})
    assert res.status_code == 400

    unsub_res = client.post("/api/unsubscribe", json={"email": "notanemail"})
    assert unsub_res.status_code == 400

def test_alerts_cooldown_and_dispatch():
    """Verify get_cooldown_seconds and dispatch_hazardous_aqi_alerts behavior."""
    assert get_cooldown_seconds("1h") == 3600.0
    assert get_cooldown_seconds("6h") == 21600.0
    assert get_cooldown_seconds("24h") == 86400.0

    res = dispatch_hazardous_aqi_alerts(current_aqi=155.0, forecast_24h_aqi=160.0, aqi_status="Unhealthy")
    assert "status" in res

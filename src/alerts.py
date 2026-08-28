import os
import json
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER, LOCATION_NAME
except ImportError:
    from src.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER, LOCATION_NAME

# In-memory cooldown tracking (timestamp of last sent alert)
_LAST_ALERT_TIME = 0.0
ALERT_COOLDOWN_SECONDS = 3600  # 1 hour cooldown between email blasts

def dispatch_hazardous_aqi_alerts(current_aqi: float, forecast_24h_aqi: float, aqi_status: str) -> dict:
    """
    Checks subscriber list and dispatches automated email alerts if AQI exceeds Unhealthy threshold (>150).
    Includes cooldown logic and non-blocking diagnostic logging.
    """
    global _LAST_ALERT_TIME

    # Only trigger if current or 24h forecast exceeds Unhealthy threshold (AQI > 150)
    max_aqi = max(current_aqi, forecast_24h_aqi)
    if max_aqi <= 150.0:
        return {"status": "skipped", "reason": "AQI level below hazardous threshold (<= 150)"}

    # Check cooldown window
    now = time.time()
    if (now - _LAST_ALERT_TIME) < ALERT_COOLDOWN_SECONDS:
        remaining_min = int((ALERT_COOLDOWN_SECONDS - (now - _LAST_ALERT_TIME)) / 60)
        return {"status": "cooldown", "reason": f"Alert cooldown active ({remaining_min} mins remaining)"}

    sub_file = os.path.join("data", "subscribers.json")
    if not os.path.exists(sub_file):
        return {"status": "skipped", "reason": "No subscriber database found"}

    try:
        with open(sub_file, "r") as f:
            subscribers = json.load(f)
    except Exception as e:
        return {"status": "error", "reason": f"Failed to read subscribers.json: {e}"}

    if not subscribers:
        return {"status": "skipped", "reason": "Subscriber list is empty"}

    # Diagnostic check for SMTP server credentials
    if not (SMTP_HOST and SMTP_USER and SMTP_PASSWORD):
        print(f"⚠️ [Alert Dispatcher] Hazardous AQI ({max_aqi:.1f}) detected for {LOCATION_NAME}!")
        print(f"   SMTP credentials (SMTP_HOST, SMTP_USER) not set in .env.")
        print(f"   {len(subscribers)} registered subscribers queued for notifications: {subscribers}")
        _LAST_ALERT_TIME = now
        return {
            "status": "dry_run",
            "message": "Hazardous AQI alert logged. SMTP credentials unconfigured in environment.",
            "subscribers_queued": len(subscribers)
        }

    # Build Email Message
    subject = f"⚠️ HAZARDOUS AQI ALERT: {LOCATION_NAME} Air Quality Reached {current_aqi:.1f} ({aqi_status})"
    body = (
        f"PEARLS AQI Intelligence System Advisory\n"
        f"----------------------------------------\n"
        f"Location: {LOCATION_NAME}\n"
        f"Current Observed AQI: {current_aqi:.1f} ({aqi_status})\n"
        f"24-Hour Forecasted AQI: {forecast_24h_aqi:.1f}\n\n"
        f"Health Advisory:\n"
        f"Everyone may begin to experience health effects. Members of sensitive groups "
        f"should avoid all outdoor physical activity.\n\n"
        f"Stay safe,\nPEARLS MLOps Automated Monitoring System"
    )

    sent_count = 0
    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10.0)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)

        for recipient in subscribers:
            msg = MIMEMultipart()
            msg["From"] = SMTP_SENDER
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))
            
            server.sendmail(SMTP_SENDER, [recipient], msg.as_string())
            sent_count += 1

        server.quit()
        _LAST_ALERT_TIME = now
        print(f"✅ [Alert Dispatcher] Dispatched hazardous AQI email alerts to {sent_count} subscribers!")
        return {"status": "success", "emails_sent": sent_count}
    except Exception as e:
        print(f"❌ [Alert Dispatcher] Failed to send SMTP emails: {e}")
        return {"status": "error", "reason": str(e)}

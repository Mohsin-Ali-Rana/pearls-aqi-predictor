import os
import json
import time
import socket
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr

try:
    from config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER_NAME, LOCATION_NAME
except ImportError:
    from src.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_SENDER_NAME, LOCATION_NAME


def connect_smtp_server(host: str, port: int = 587, timeout: float = 3.0):
    """
    Establishes direct SMTP / SMTPS connection using host domain ('smtp.gmail.com')
    over IPv4 (AF_INET) to prevent '[Errno 101] Network is unreachable' in cloud containers
    without IPv6 egress routing.
    """
    orig_getaddrinfo = socket.getaddrinfo

    def ipv4_only_getaddrinfo(h, p, family=0, type=0, proto=0, flags=0):
        res = orig_getaddrinfo(h, p, family, type, proto, flags)
        ipv4_res = [r for r in res if r[0] == socket.AF_INET]
        return ipv4_res if ipv4_res else res

    last_err = None
    ports_to_try = [port]
    alt_port = 465 if port != 465 else 587
    if alt_port not in ports_to_try:
        ports_to_try.append(alt_port)

    socket.getaddrinfo = ipv4_only_getaddrinfo
    try:
        for p in ports_to_try:
            try:
                if p == 465:
                    server = smtplib.SMTP_SSL(host, p, timeout=timeout)
                else:
                    server = smtplib.SMTP(host, p, timeout=timeout)
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                print(f"[SMTP Connect] Successfully established IPv4 SMTP connection to {host} on port {p}")
                return server
            except Exception as err:
                print(f"[SMTP Connect] Connection attempt failed on {host}:{p} ({err})")
                last_err = err

        if last_err:
            raise last_err
        raise RuntimeError(f"Unable to establish SMTP connection to {host}")
    finally:
        socket.getaddrinfo = orig_getaddrinfo

# In-memory cooldown tracking (timestamp of last sent alert)
_LAST_ALERT_TIME = 0.0
ALERT_COOLDOWN_SECONDS = 3600  # 1 hour cooldown between email blasts


# User preference ke hisaab se cooldown duration in seconds calculate karne ka function
def get_cooldown_seconds(freq_str: str) -> float:
    if freq_str == "24h" or freq_str == "daily":
        return 86400.0  # 24 hours
    elif freq_str == "6h":
        return 21600.0  # 6 hours
    elif freq_str == "1h" or freq_str == "hourly":
        return 3600.0   # 1 hour
    return 21600.0      # 6 hours default






# Welcome aur subscription update email ke liye executive HTML layout generate karne ka builder
def build_welcome_html_email(recipient_email: str, location_name: str, threshold: int, frequency: str, is_update: bool = False) -> str:
    freq_label = "Max 1 alert per 6 hours" if frequency == "6h" else ("Max 1 alert per day" if frequency == "24h" else "Max 1 alert per hour")
    header_title = "Preferences Updated" if is_update else "Subscription Confirmed"
    intro_text = (
        "Your alert threshold and notification frequency preferences for the PEARLS AQI Early Warning Network have been successfully updated."
        if is_update
        else "Welcome to the PEARLS AQI Early Warning Network. Your email address has been successfully registered to receive real-time automated atmospheric alerts."
    )

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{header_title} - PEARLS AQI Intelligence</title>
</head>
<body style="margin:0; padding:0; background-color:#f1f5f9; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing:antialiased; color:#0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f1f5f9; padding:40px 10px;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" style="max-width:600px; background-color:#ffffff; border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; box-shadow:0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                    
                    <!-- Executive Header -->
                    <tr>
                        <td style="background-color:#0f172a; padding:32px 36px; text-align:left; border-bottom:3px solid #0d9488;">
                            <div style="font-size:11px; font-weight:800; color:#0d9488; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;">
                                PEARLS AQI ATMOSPHERIC INTELLIGENCE
                            </div>
                            <h1 style="font-size:22px; font-weight:700; color:#ffffff; margin:0 0 6px 0; letter-spacing:-0.5px;">
                                {header_title}
                            </h1>
                            <div style="font-size:13px; color:#94a3b8; margin:0;">
                                Automated Monitoring & Early Warning System
                            </div>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding:36px; background-color:#ffffff;">
                            <p style="font-size:15px; line-height:1.6; color:#334155; margin-top:0; margin-bottom:24px;">
                                {intro_text}
                            </p>

                            <!-- Configuration Table Card -->
                            <div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:20px; margin-bottom:28px;">
                                <div style="font-size:12px; font-weight:700; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:14px; border-bottom:1px solid #e2e8f0; padding-bottom:8px;">
                                    Active Notification Parameters
                                </div>
                                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:14px; line-height:1.8;">
                                    <tr>
                                        <td style="color:#64748b; font-weight:500; width:40%;">Target Subscriber:</td>
                                        <td style="color:#0f172a; font-weight:600;">{recipient_email}</td>
                                    </tr>
                                    <tr>
                                        <td style="color:#64748b; font-weight:500;">Monitoring Region:</td>
                                        <td style="color:#0f172a; font-weight:600;">{location_name}</td>
                                    </tr>
                                    <tr>
                                        <td style="color:#64748b; font-weight:500;">Alert Threshold:</td>
                                        <td style="color:#0f172a; font-weight:600;">AQI &gt; {threshold}</td>
                                    </tr>
                                    <tr>
                                        <td style="color:#64748b; font-weight:500;">Max Frequency:</td>
                                        <td style="color:#0f172a; font-weight:600;">{freq_label}</td>
                                    </tr>
                                </table>
                            </div>

                            <p style="font-size:14px; line-height:1.6; color:#475569; margin-bottom:0;">
                                You will receive an immediate notification whenever our live predictive inference engine detects or forecasts air quality levels breaching your designated threshold.
                            </p>
                        </td>
                    </tr>

                    <!-- Executive Footer -->
                    <tr>
                        <td style="background-color:#f8fafc; padding:24px 36px; border-top:1px solid #e2e8f0; text-align:left;">
                            <div style="font-size:13px; font-weight:600; color:#334155; margin-bottom:4px;">
                                PEARLS MLOps Engineering Team
                            </div>
                            <div style="font-size:12px; color:#64748b; line-height:1.5;">
                                Central Atmospheric Observation Station &bull; Feature Store Integration<br>
                                This is an automated operational dispatch. Please do not reply directly to this email.
                            </div>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""






# Hazardous AQI threshold breach email alerts ke liye HTML layout builder
def build_alert_html_email(recipient_email: str, location_name: str, current_aqi: float, forecast_24h_aqi: float, aqi_status: str, threshold: int) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Air Quality Alert - PEARLS AQI Intelligence</title>
</head>
<body style="margin:0; padding:0; background-color:#f1f5f9; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing:antialiased; color:#0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f1f5f9; padding:40px 10px;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" style="max-width:600px; background-color:#ffffff; border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; box-shadow:0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                    
                    <!-- Executive Header -->
                    <tr>
                        <td style="background-color:#991b1b; padding:32px 36px; text-align:left; border-bottom:3px solid #7f1d1d;">
                            <div style="font-size:11px; font-weight:800; color:#fca5a5; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;">
                                HAZARDOUS AIR QUALITY ADVISORY
                            </div>
                            <h1 style="font-size:22px; font-weight:700; color:#ffffff; margin:0 0 6px 0; letter-spacing:-0.5px;">
                                AQI Threshold Exceeded
                            </h1>
                            <div style="font-size:13px; color:#fecaca; margin:0;">
                                Region: {location_name}
                            </div>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding:36px; background-color:#ffffff;">
                            <p style="font-size:15px; line-height:1.6; color:#334155; margin-top:0; margin-bottom:24px;">
                                The atmospheric monitoring engine has recorded an Air Quality Index (AQI) reading that breaches your designated safety threshold (AQI &gt; {threshold}).
                            </p>

                            <!-- Metric Card -->
                            <div style="background-color:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:20px; margin-bottom:28px;">
                                <div style="font-size:12px; font-weight:700; color:#991b1b; text-transform:uppercase; letter-spacing:1px; margin-bottom:14px; border-bottom:1px solid #fca5a5; padding-bottom:8px;">
                                    Telemetry Summary
                                </div>
                                <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="font-size:14px; line-height:1.8;">
                                    <tr>
                                        <td style="color:#7f1d1d; font-weight:500; width:45%;">Current Observed AQI:</td>
                                        <td style="color:#991b1b; font-weight:700; font-size:16px;">{current_aqi:.1f} ({aqi_status})</td>
                                    </tr>
                                    <tr>
                                        <td style="color:#7f1d1d; font-weight:500;">24-Hour Forecast AQI:</td>
                                        <td style="color:#991b1b; font-weight:700; font-size:16px;">{forecast_24h_aqi:.1f}</td>
                                    </tr>
                                    <tr>
                                        <td style="color:#7f1d1d; font-weight:500;">Configured Threshold:</td>
                                        <td style="color:#0f172a; font-weight:600;">AQI &gt; {threshold}</td>
                                    </tr>
                                </table>
                            </div>

                            <div style="background-color:#f8fafc; border-left:4px solid #0d9488; padding:16px; border-radius:0 8px 8px 0; margin-bottom:24px;">
                                <div style="font-size:13px; font-weight:700; color:#0f172a; margin-bottom:4px;">Recommended Precautionary Actions</div>
                                <div style="font-size:13px; color:#475569; line-height:1.5;">
                                    Consider limiting prolonged outdoor exposure and keep indoor ventilation systems active until air quality readings return below safety thresholds.
                                </div>
                            </div>
                        </td>
                    </tr>

                    <!-- Executive Footer -->
                    <tr>
                        <td style="background-color:#f8fafc; padding:24px 36px; border-top:1px solid #e2e8f0; text-align:left;">
                            <div style="font-size:13px; font-weight:600; color:#334155; margin-bottom:4px;">
                                PEARLS MLOps Engineering Team
                            </div>
                            <div style="font-size:12px; color:#64748b; line-height:1.5;">
                                Central Atmospheric Observation Station &bull; Real-Time Inference System<br>
                                You are receiving this advisory based on your subscriber preferences.
                            </div>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""






# Subscriber threshold check karke automated SMTP email alert dispatch karne ka function
def dispatch_hazardous_aqi_alerts(current_aqi: float, forecast_24h_aqi: float, aqi_status: str) -> dict:
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

    max_aqi = max(current_aqi, forecast_24h_aqi)
    now = time.time()
    eligible_subscribers = []
    updated_subscribers = []

    for sub in subscribers:
        if isinstance(sub, str):
            email = sub.strip().lower()
            thresh = 100.0
            freq = "6h"
            last_sent = 0.0
        else:
            email = str(sub.get("email", "")).strip().lower()
            try:
                thresh = float(sub.get("threshold", 100))
            except (ValueError, TypeError):
                thresh = 100.0
            freq = str(sub.get("frequency", "6h"))
            try:
                last_sent = float(sub.get("last_sent", 0.0))
            except (ValueError, TypeError):
                last_sent = 0.0

        if not email:
            continue

        cooldown_sec = get_cooldown_seconds(freq)

        if max_aqi >= thresh and (now - last_sent) >= cooldown_sec:
            eligible_subscribers.append((email, thresh))
            if isinstance(sub, dict):
                sub["last_sent"] = now
                updated_subscribers.append(sub)
            else:
                updated_subscribers.append({"email": email, "threshold": int(thresh), "frequency": freq, "last_sent": now})
        else:
            updated_subscribers.append(sub)

    if not eligible_subscribers:
        return {"status": "skipped", "reason": "No subscribers meet threshold or cooldown conditions"}

    host, port, user, password, sender_name = get_smtp_config()

    if not (host and user and password):
        print(f"[Alert Dispatcher] Hazardous AQI ({max_aqi:.1f}) detected for {LOCATION_NAME}!")
        with open(sub_file, "w") as f:
            json.dump(updated_subscribers, f, indent=2)
        return {
            "status": "dry_run",
            "message": "Hazardous AQI alert logged. SMTP credentials unconfigured in environment.",
            "subscribers_queued": len(eligible_subscribers)
        }

    subject = f"AQI Threshold Alert: {LOCATION_NAME} - Observed AQI {current_aqi:.1f} ({aqi_status})"

    # Method 1: Resend HTTP API (HTTPS Port 443 - Bypasses Railway Egress Firewall)
    resend_api_key = os.getenv("RESEND_API_KEY")
    if resend_api_key:
        sent_count = 0
        for recipient, thresh in eligible_subscribers:
            plain_body = (
                f"PEARLS AQI Intelligence System Advisory\n"
                f"Location: {LOCATION_NAME}\n"
                f"Current Observed AQI: {current_aqi:.1f} ({aqi_status})\n"
                f"24-Hour Forecast AQI: {forecast_24h_aqi:.1f}\n"
                f"Configured Safety Threshold: AQI > {thresh}\n\n"
                f"Please take appropriate health safety precautions.\n"
                f"PEARLS MLOps Engineering Team"
            )
            html_body = build_alert_html_email(recipient, LOCATION_NAME, current_aqi, forecast_24h_aqi, aqi_status, thresh)
            res = send_email_via_resend(resend_api_key, recipient, subject, html_body, plain_body, sender_name)
            if res.get("status") == "success":
                sent_count += 1

        with open(sub_file, "w") as f:
            json.dump(updated_subscribers, f, indent=2)

        print(f"[Alert Dispatcher] Dispatched custom AQI email alerts via Resend to {sent_count} subscribers!")
        return {"status": "success", "emails_sent": sent_count}

    # Method 2: Standard Direct SMTP
    if not (host and user and password):
        print(f"[Alert Dispatcher] Hazardous AQI ({max_aqi:.1f}) detected for {LOCATION_NAME}!")
        with open(sub_file, "w") as f:
            json.dump(updated_subscribers, f, indent=2)
        return {
            "status": "dry_run",
            "message": "Hazardous AQI alert logged. SMTP credentials unconfigured in environment.",
            "subscribers_queued": len(eligible_subscribers)
        }

    sent_count = 0
    try:
        server = connect_smtp_server(host, port, timeout=12.0)
        server.login(user, password)

        sender_header = formataddr((sender_name, user))

        for recipient, thresh in eligible_subscribers:
            msg = MIMEMultipart("alternative")
            msg["From"] = sender_header
            msg["To"] = recipient
            msg["Reply-To"] = user
            msg["Subject"] = subject
            msg["X-Mailer"] = "PEARLS-AQI-Predictor/2.0"
            msg["Auto-Submitted"] = "auto-generated"
            
            plain_body = (
                f"PEARLS AQI Intelligence System Advisory\n"
                f"Location: {LOCATION_NAME}\n"
                f"Current Observed AQI: {current_aqi:.1f} ({aqi_status})\n"
                f"24-Hour Forecast AQI: {forecast_24h_aqi:.1f}\n"
                f"Configured Safety Threshold: AQI > {thresh}\n\n"
                f"Please take appropriate health safety precautions.\n"
                f"PEARLS MLOps Engineering Team"
            )
            html_body = build_alert_html_email(recipient, LOCATION_NAME, current_aqi, forecast_24h_aqi, aqi_status, thresh)

            msg.attach(MIMEText(plain_body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
            
            server.sendmail(user, [recipient], msg.as_string())
            sent_count += 1

        server.quit()

        with open(sub_file, "w") as f:
            json.dump(updated_subscribers, f, indent=2)

        print(f"[Alert Dispatcher] Dispatched custom AQI email alerts to {sent_count} subscribers!")
        return {"status": "success", "emails_sent": sent_count}
    except Exception as e:
        print(f"[Alert Dispatcher] Failed to send SMTP emails: {e}")
        return {"status": "error", "reason": str(e)}






def get_smtp_config():
    host = os.getenv("SMTP_HOST") or SMTP_HOST or "smtp.gmail.com"
    port_val = os.getenv("SMTP_PORT") or SMTP_PORT or 587
    try:
        port = int(port_val)
    except (ValueError, TypeError):
        port = 587
    user = os.getenv("SMTP_USER") or SMTP_USER or ""
    password = os.getenv("SMTP_PASSWORD") or SMTP_PASSWORD or ""
    sender_name = os.getenv("SMTP_SENDER_NAME") or SMTP_SENDER_NAME or "Pearls AQI Intelligence"
    return host, port, user, password, sender_name


def send_email_via_resend(api_key: str, recipient: str, subject: str, html_body: str, plain_body: str, sender_name: str) -> dict:
    """
    Dispatches emails via Resend HTTP API over standard HTTPS Port 443.
    Bypasses cloud firewall restrictions on outbound SMTP ports (25, 465, 587).
    """
    import urllib.request
    import urllib.error
    import json

    url = "https://api.resend.com/emails"
    clean_key = api_key.strip()
    headers = {
        "Authorization": f"Bearer {clean_key}",
        "Content-Type": "application/json"
    }

    # Resend free tier default sender domain
    from_address = os.getenv("RESEND_FROM") or "onboarding@resend.dev"

    payload = {
        "from": from_address,
        "to": [recipient],
        "subject": subject,
        "html": html_body,
        "text": plain_body
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=10.0) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            print(f"[Resend HTTP API] Dispatched email to {recipient} (ID: {res_data.get('id')})")
            return {"status": "success", "message": f"Email sent via Resend HTTP API to {recipient}", "id": res_data.get("id")}
    except urllib.error.HTTPError as http_err:
        err_body = http_err.read().decode("utf-8", errors="ignore")
        print(f"[Resend HTTP API] HTTP {http_err.code} Error for {recipient}: {err_body}")
        try:
            err_json = json.loads(err_body)
            msg = err_json.get("message") or err_body
        except Exception:
            msg = err_body
        return {"status": "error", "reason": f"Resend API HTTP {http_err.code}: {msg}"}
    except Exception as e:
        print(f"[Resend HTTP API] Error sending email to {recipient}: {e}")
        return {"status": "error", "reason": f"Resend HTTP API error: {e}"}


# Subscribed new user ko welcome email send karne ka handler
def send_welcome_email(recipient_email: str, threshold: int = 100, frequency: str = "6h", is_update: bool = False) -> dict:
    host, port, user, password, sender_name = get_smtp_config()

    freq_label = "Max 1 alert per 6 hours" if frequency == "6h" else ("Max 1 alert per day" if frequency == "24h" else "Max 1 alert per hour")
    subject = f"Subscription Preferences Updated: PEARLS AQI Automated Alert Dispatcher" if is_update else f"Subscription Confirmation: PEARLS AQI Automated Alert Dispatcher"
    
    plain_body = (
        f"PEARLS AQI Intelligence System Advisory\n\n"
        f"Your email ({recipient_email}) preferences have been {'updated' if is_update else 'registered'} for {LOCATION_NAME}.\n"
        f"Alert Threshold: AQI > {threshold}\n"
        f"Max Frequency: {freq_label}\n\n"
        f"Best regards,\nPEARLS MLOps Engineering Team"
    )

    html_body = build_welcome_html_email(recipient_email, LOCATION_NAME, threshold, frequency, is_update=is_update)

    # Method 1: Resend HTTP API (HTTPS Port 443)
    resend_api_key = os.getenv("RESEND_API_KEY")
    if resend_api_key:
        return send_email_via_resend(resend_api_key, recipient_email, subject, html_body, plain_body, sender_name)

    # Method 2: Standard Direct SMTP
    if not (host and user and password):
        print(f"[Welcome Email] SMTP credentials unconfigured. Dry-run mode for {recipient_email}.")
        return {"status": "dry_run", "message": "SMTP credentials unconfigured on backend environment."}

    try:
        server = connect_smtp_server(host, port, timeout=12.0)
        server.login(user, password)

        sender_header = formataddr((sender_name, user))
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_header
        msg["To"] = recipient_email
        msg["Reply-To"] = user
        msg["Subject"] = subject
        msg["X-Mailer"] = "PEARLS-AQI-Predictor/2.0"
        msg["Auto-Submitted"] = "auto-generated"

        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        server.sendmail(user, [recipient_email], msg.as_string())
        server.quit()
        print(f"[Welcome Email] Successfully dispatched executive HTML email to {recipient_email}!")
        return {"status": "success", "message": f"Executive welcome email sent to {recipient_email}"}
    except Exception as e:
        print(f"[Welcome Email] SMTP Error sending to {recipient_email}: {e}")
        return {"status": "error", "reason": str(e)}






# Unsubscribe confirmation email ke liye HTML builder layout
def build_unsubscribe_html_email(recipient_email: str, location_name: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unsubscribed - PEARLS AQI Intelligence</title>
</head>
<body style="margin:0; padding:0; background-color:#f1f5f9; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color:#0f172a;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background-color:#f1f5f9; padding:40px 10px;">
        <tr>
            <td align="center">
                <table role="presentation" width="100%" style="max-width:600px; background-color:#ffffff; border-radius:12px; overflow:hidden; border:1px solid #e2e8f0; box-shadow:0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background-color:#475569; padding:32px 36px; text-align:left; border-bottom:3px solid #334155;">
                            <div style="font-size:11px; font-weight:800; color:#cbd5e1; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px;">
                                PEARLS AQI ATMOSPHERIC INTELLIGENCE
                            </div>
                            <h1 style="font-size:22px; font-weight:700; color:#ffffff; margin:0 0 6px 0; letter-spacing:-0.5px;">
                                Unsubscription Confirmed
                            </h1>
                            <div style="font-size:13px; color:#94a3b8; margin:0;">
                                Region: {location_name}
                            </div>
                        </td>
                    </tr>

                    <!-- Body Content -->
                    <tr>
                        <td style="padding:36px; background-color:#ffffff;">
                            <p style="font-size:15px; line-height:1.6; color:#334155; margin-top:0; margin-bottom:20px;">
                                Your email address (<strong>{recipient_email}</strong>) has been successfully unsubscribed from the PEARLS AQI Early Warning Network.
                            </p>
                            <p style="font-size:14px; line-height:1.6; color:#64748b; margin-bottom:24px;">
                                You will no longer receive automated hazardous AQI notifications or threshold advisories for {location_name}.
                            </p>
                            <div style="background-color:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px; font-size:13px; color:#475569; line-height:1.5;">
                                If you did not request this unsubscription or wish to re-enable alerts in the future, you can resubscribe anytime directly through the PEARLS AQI Command Center dashboard.
                            </div>
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="background-color:#f8fafc; padding:24px 36px; border-top:1px solid #e2e8f0; text-align:left;">
                            <div style="font-size:13px; font-weight:600; color:#334155; margin-bottom:4px;">
                                PEARLS MLOps Engineering Team
                            </div>
                            <div style="font-size:12px; color:#64748b; line-height:1.5;">
                                Central Atmospheric Observation Station &bull; Feature Store Integration
                            </div>
                        </td>
                    </tr>

                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""






# Unsubscribe user ko confirmation email send karne ka handler
def send_unsubscribe_email(recipient_email: str) -> dict:
    host, port, user, password, sender_name = get_smtp_config()

    subject = f"Unsubscription Confirmed: PEARLS AQI Automated Alert Dispatcher"
    
    plain_body = (
        f"PEARLS AQI Intelligence System Advisory\n\n"
        f"Your email ({recipient_email}) has been successfully unsubscribed from the PEARLS AQI alert system for {LOCATION_NAME}.\n"
        f"You will no longer receive automated notification dispatches.\n\n"
        f"Best regards,\nPEARLS MLOps Engineering Team"
    )

    html_body = build_unsubscribe_html_email(recipient_email, LOCATION_NAME)

    # Method 1: Resend HTTP API (HTTPS Port 443)
    resend_api_key = os.getenv("RESEND_API_KEY")
    if resend_api_key:
        return send_email_via_resend(resend_api_key, recipient_email, subject, html_body, plain_body, sender_name)

    # Method 2: Standard Direct SMTP
    if not (host and user and password):
        print(f"[Unsubscribe Email] SMTP credentials unconfigured. Dry-run mode for {recipient_email}.")
        return {"status": "dry_run", "message": "SMTP credentials unconfigured on backend environment."}

    try:
        server = connect_smtp_server(host, port, timeout=12.0)
        server.login(user, password)

        sender_header = formataddr((sender_name, user))
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_header
        msg["To"] = recipient_email
        msg["Reply-To"] = user
        msg["Subject"] = subject
        msg["X-Mailer"] = "PEARLS-AQI-Predictor/2.0"
        msg["Auto-Submitted"] = "auto-generated"

        msg.attach(MIMEText(plain_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        server.sendmail(user, [recipient_email], msg.as_string())
        server.quit()
        print(f"[Unsubscribe Email] Dispatched unsubscription confirmation email to {recipient_email}!")
        return {"status": "success", "message": f"Unsubscribe confirmation email sent to {recipient_email}"}
    except Exception as e:
        print(f"[Unsubscribe Email] SMTP Error sending to {recipient_email}: {e}")
        return {"status": "error", "reason": str(e)}

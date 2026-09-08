import os
from pathlib import Path
from dotenv import load_dotenv

# Project ki root path find karke .env file load kar rahe hain
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)






# Hopsworks feature store ke connection settings
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "MA")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", 443))






# Check kar rahe hain ke Hopsworks key mojood hai ya nahi
def validate_credentials():
    if not HOPSWORKS_API_KEY:
        raise EnvironmentError("HOPSWORKS_API_KEY is not configured in .env or system environment.")
    return True






# Wah Cantt / Taxila region ke location details aur coordinates
LOCATION_LATITUDE = 33.77
LOCATION_LONGITUDE = 72.75
LOCATION_NAME = os.getenv("LOCATION_NAME", "Wah Cantt / Taxila Region")
STATION_NAME = os.getenv("STATION_NAME", "Central Atmospheric Observation Station")






# Open-Meteo API ka endpoint URL
OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"






# Email alerts ke liye SMTP configurations
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER_NAME = os.getenv("SMTP_SENDER_NAME", "Pearls AQI Intelligence")
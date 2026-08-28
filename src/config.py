import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at project root
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Hopsworks Credentials & Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
if not HOPSWORKS_API_KEY:
    raise EnvironmentError("HOPSWORKS_API_KEY is not configured in .env or system environment.")

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "MA")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", 443))

# Geographical Location & Station Metadata
LOCATION_LATITUDE = 33.77
LOCATION_LONGITUDE = 72.75
LOCATION_NAME = os.getenv("LOCATION_NAME", "Wah Cantt / Taxila Region")
STATION_NAME = os.getenv("STATION_NAME", "Central Atmospheric Observation Station")

# Open-Meteo Air Quality API Endpoint
OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# SMTP Email Alert Credentials
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", SMTP_USER)
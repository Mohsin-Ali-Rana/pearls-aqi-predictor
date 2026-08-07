import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at project root
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

# Hopsworks Credentials & Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "MA")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
HOPSWORKS_PORT = int(os.getenv("HOPSWORKS_PORT", 443))

# Geographical Coordinates for the Wah/Taxila region
LOCATION_LATITUDE = 33.77
LOCATION_LONGITUDE = 72.75

# Open-Meteo Air Quality API Endpoint
OPEN_METEO_AQI_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
# OPEN_METEO_AQI_URL = "https://archive-api.open-meteo.com/v1/archive"
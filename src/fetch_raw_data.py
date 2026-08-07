import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    from config import LOCATION_LATITUDE, LOCATION_LONGITUDE, OPEN_METEO_AQI_URL
except ImportError:
    from src.config import LOCATION_LATITUDE, LOCATION_LONGITUDE, OPEN_METEO_AQI_URL

def fetch_historical_aqi(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical hourly air quality data from Open-Meteo and returns a Pandas DataFrame.
    Includes explicit timeouts and exponential backoff retry strategy.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "european_aqi"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    print(f"Pinging Open-Meteo API for coordinates ({lat}, {lon})...")
    response = session.get(OPEN_METEO_AQI_URL, params=params, timeout=(5.0, 25.0))
    response.raise_for_status() 
    
    data = response.json()
    hourly_data = data.get("hourly", {})
    
    df = pd.DataFrame(hourly_data)
    if not df.empty and 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    return df

if __name__ == "__main__":
    # Test the pipeline by fetching a 30-day historical window
    df_raw = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, "2024-07-01", "2026-07-21")
    
    print("\n--- Data Preview ---")
    print(df_raw.head())
    
    print("\n--- Data Schema & Missing Values ---")
    print(df_raw.info())
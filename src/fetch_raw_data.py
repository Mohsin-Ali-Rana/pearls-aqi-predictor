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
    Fetches historical hourly air quality and meteorological data from Open-Meteo APIs.
    Includes explicit timeouts and exponential backoff retry strategy.
    """
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "european_aqi"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    print(f"Pinging Open-Meteo AQI API for coordinates ({lat}, {lon})...")
    res_aqi = session.get(OPEN_METEO_AQI_URL, params=aqi_params, timeout=(5.0, 25.0))
    res_aqi.raise_for_status()
    df_aqi = pd.DataFrame(res_aqi.json().get("hourly", {}))

    print(f"Pinging Open-Meteo Weather API for coordinates ({lat}, {lon})...")
    try:
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        res_weather = session.get(weather_url, params=weather_params, timeout=(5.0, 25.0))
        res_weather.raise_for_status()
    except Exception:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        res_weather = session.get(weather_url, params=weather_params, timeout=(5.0, 25.0))
        res_weather.raise_for_status()

    df_weather = pd.DataFrame(res_weather.json().get("hourly", {}))

    if not df_aqi.empty and 'time' in df_aqi.columns and not df_weather.empty and 'time' in df_weather.columns:
        df_aqi['time'] = pd.to_datetime(df_aqi['time'])
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df = pd.merge(df_aqi, df_weather, on='time', how='inner')
    elif not df_aqi.empty:
        df = df_aqi
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
    else:
        df = pd.DataFrame()
    
    return df

if __name__ == "__main__":
    from datetime import datetime, timedelta
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_str = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
    df_raw = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, start_str, today_str)
    
    print("\n--- Data Preview ---")
    print(df_raw.head())
    
    print("\n--- Data Schema & Missing Values ---")
    print(df_raw.info())
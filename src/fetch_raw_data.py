import os
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

try:
    from config import LOCATION_LATITUDE, LOCATION_LONGITUDE, OPEN_METEO_AQI_URL
except ImportError:
    from src.config import LOCATION_LATITUDE, LOCATION_LONGITUDE, OPEN_METEO_AQI_URL


# --- Stage: Data Ingestion ---
# Open-Meteo API se AQI aur weather data fetch karne ka main function
def fetch_historical_aqi(lat: float, lon: float, start_date: str = None, end_date: str = None, days: int = 14) -> pd.DataFrame:
    # Agar date range specify nahi ki tou by default pichle 'days' ka range calculate hoga
    if not end_date:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    # API calls ke liye retry strategy setup kar rahe hain taake network timeout handle ho sake
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "european_aqi"],
        "timezone": "auto"
    }
    
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure"],
        "timezone": "auto"
    }

    print(f"Fetching Open-Meteo AQI data ({start_date} to {end_date}) for coordinates ({lat}, {lon})...")
    res_aqi = session.get(OPEN_METEO_AQI_URL, params=aqi_params, timeout=(10.0, 60.0))
    res_aqi.raise_for_status()
    df_aqi = pd.DataFrame(res_aqi.json().get("hourly", {}))

    print(f"Fetching Open-Meteo Weather data ({start_date} to {end_date}) for coordinates ({lat}, {lon})...")
    weather_url_archive = "https://archive-api.open-meteo.com/v1/archive"
    try:
        res_weather = session.get(weather_url_archive, params=weather_params, timeout=(10.0, 60.0))
        res_weather.raise_for_status()
    except Exception as e:
        print(f"Weather archive API unavailable ({e}), falling back to forecast endpoint...")
        weather_url_forecast = "https://api.open-meteo.com/v1/forecast"
        res_weather = session.get(weather_url_forecast, params=weather_params, timeout=(10.0, 60.0))
        res_weather.raise_for_status()

    df_weather = pd.DataFrame(res_weather.json().get("hourly", {}))

    # Both DataFrames ko timestamp par merge kar rahe hain
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

    df = df.sort_values('time').reset_index(drop=True)
    return df






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo Air Quality & Weather Data")
    parser.add_argument("--days", type=int, default=14, help="Number of historical days to fetch (default: 14)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    print(f"Starting raw data ingestion (days={args.days})...")
    df_raw = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, start_date=args.start_date, end_date=args.end_date, days=args.days)
    
    os.makedirs("data", exist_ok=True)
    raw_path = os.path.join("data", "raw_aqi.csv")
    df_raw.to_csv(raw_path, index=False)
    
    row_count = len(df_raw)
    print(f"Successfully saved {row_count} hourly observation records to '{raw_path}'.")
    
    print("\n--- Data Summary ---")
    print(f"Start Date: {df_raw['time'].min()} | End Date: {df_raw['time'].max()} | Total Rows: {row_count}")
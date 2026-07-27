import requests
import pandas as pd
from config import LOCATION_LATITUDE, LOCATION_LONGITUDE, OPEN_METEO_AQI_URL

def fetch_historical_aqi(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical hourly air quality data from Open-Meteo and returns a Pandas DataFrame.
    """
    # 1. Define the API parameters
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm10", "pm2_5", "nitrogen_dioxide", "ozone", "european_aqi"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto"
    }
    
    # 2. Execute the HTTP GET request
    print(f"Pinging Open-Meteo API for coordinates ({lat}, {lon})...")
    response = requests.get(OPEN_METEO_AQI_URL, params=params)
    
    # 3. Exception Handling: Safely crash if the API rejects our request
    response.raise_for_status() 
    
    # 4. Parse the JSON payload
    data = response.json()
    hourly_data = data.get("hourly", {})
    
    # 5. Transform into a structured DataFrame
    df = pd.DataFrame(hourly_data)
    
    # 6. Type Casting: Convert the raw time string into a mathematical datetime object
    df['time'] = pd.to_datetime(df['time'])
    
    return df

if __name__ == "__main__":
    # Test the pipeline by fetching a 30-day historical window
    df_raw = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, "2024-07-01", "2026-07-21")
    
    print("\n--- Data Preview ---")
    print(df_raw.head())
    
    print("\n--- Data Schema & Missing Values ---")
    print(df_raw.info())
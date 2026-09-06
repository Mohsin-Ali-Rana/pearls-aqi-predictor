import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import hopsworks

try:
    from fetch_raw_data import fetch_historical_aqi
    from config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )
except ImportError:
    from src.fetch_raw_data import fetch_historical_aqi
    from src.config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )


# --- Stage: Feature Engineering ---
# Raw observation data se cyclical signals, lags aur rolling features generate kar rahe hain
def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. Cyclical Calendar Features (Hour aur day of week ko sine/cosine mein encode kar rahe hain)
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['is_weekend'] = df['time'].dt.dayofweek.isin([5, 6]).astype(int)
    
    df['sin_hour'] = np.sin(2 * np.pi * df['time'].dt.hour / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['time'].dt.hour / 24.0)
    df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    # 2. Atmospheric Dispersion Features (Wind speed aur humidity se air ventilation calculate kar rahe hain)
    if 'wind_speed_10m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['ventilation_index'] = df['wind_speed_10m'] * (100.0 - df['relative_humidity_2m']) / 100.0
        df['stagnation_index'] = ((df['relative_humidity_2m'] > 75.0) & (df['wind_speed_10m'] < 2.0)).astype(int)

    # 3. Autoregressive Lag Features (Future data leakage roknay ke liye sirf past lags t <= 0 use kiye hain)
    df['pm2_5_lag_2h'] = df['pm2_5'].shift(2)
    df['pm2_5_lag_6h'] = df['pm2_5'].shift(6)
    df['pm2_5_lag_12h'] = df['pm2_5'].shift(12)

    for col in ['pm10', 'pm2_5', 'european_aqi']:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_3h'] = df[col].shift(3)
            df[f'{col}_lag_24h'] = df[col].shift(24)
        
    # 4. Rolling Statistics (Pichle 6h, 12h, 24h ka moving average aur standard deviation)
    df['pm2_5_rolling_24h_std'] = df['pm2_5'].shift(1).rolling(window=24).std().fillna(0.0)

    for col in ['pm10', 'pm2_5', 'wind_speed_10m']:
        if col in df.columns:
            df[f'{col}_rolling_6h_mean'] = df[col].shift(1).rolling(window=6).mean()
            df[f'{col}_rolling_12h_mean'] = df[col].shift(1).rolling(window=12).mean()
            df[f'{col}_rolling_24h_mean'] = df[col].shift(1).rolling(window=24).mean()
        
    # 5. Derived Trend Features (Short-term AQI change rate)
    if 'european_aqi_lag_1h' in df.columns and 'european_aqi_lag_3h' in df.columns:
        df['aqi_change_rate'] = df['european_aqi_lag_1h'] - df['european_aqi_lag_3h']
    
    # Lagging/rolling shifting ki wajah se empty NaN rows remove kar rahe hain
    df = df.dropna().reset_index(drop=True)
    return df






# --- Stage: Feature Ingestion Pipeline ---
# Ingestion pipeline jo fresh data fetch karke Hopsworks feature group mein upload karta hai
def run_feature_pipeline():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    os.makedirs(data_dir, exist_ok=True)
    raw_path = data_dir / "raw_aqi.csv"

    print("Fetching fresh 14-day observations from Open-Meteo API...")
    fresh_df = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, days=14)
    if fresh_df.empty:
        raise ValueError("Open-Meteo API returned empty observation dataset for 14-day window.")
    fresh_df['time'] = pd.to_datetime(fresh_df['time'])

    # Existing disk raw dataset ke saath duplicate-free merge kar rahe hain
    if raw_path.exists():
        print(f"Merging fresh observations with existing records in '{raw_path}'...")
        existing_df = pd.read_csv(raw_path)
        existing_df['time'] = pd.to_datetime(existing_df['time'])
        combined_df = pd.concat([existing_df, fresh_df], ignore_index=True)
        raw_df = combined_df.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    else:
        raw_df = fresh_df.sort_values('time').reset_index(drop=True)
    
    raw_df.to_csv(raw_path, index=False)
    print(f"Updated raw dataset at '{raw_path}' ({len(raw_df)} total records).")
    
    print(f"Generating features from {len(raw_df)} observation records...")
    feature_df = generate_features(raw_df)
    if feature_df.empty:
        raise ValueError("Engineered feature set is empty. Insufficient historical observation rows.")
    feature_df["time"] = pd.to_datetime(feature_df["time"])

    print(f"Latest feature timestamp: {feature_df['time'].max()}")
    print(f"Feature dataset shape: {len(feature_df)} rows x {len(feature_df.columns)} columns.")

    # Local warm cache update kar rahe hain
    try:
        feature_df.to_parquet(data_dir / "features.parquet", index=False)
        feature_df.to_csv(data_dir / "processed_aqi.csv", index=False)
        print(f"Updated local warm cache at '{data_dir / 'features.parquet'}' ({len(feature_df)} rows).")
    except Exception as cache_err:
        print(f"Error updating local fallback cache: {cache_err}")

    # Hopsworks Feature Store synchronization
    try:
        print("Connecting to Hopsworks Feature Store...")
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host=HOPSWORKS_HOST,
            port=HOPSWORKS_PORT,
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        
        print("Uploading feature group version 2 to Hopsworks...")
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_hourly_features",
            version=2,
            primary_key=["time"],
            event_time="time",
            online_enabled=True,
            description="Hourly engineered air quality, meteorological, dispersion and cyclical time features"
        )
        
        existing_feat_names = [f.name for f in aqi_fg.features]
        new_features = []
        from hsfs.feature import Feature
        for col in feature_df.columns:
            if col not in existing_feat_names:
                dtype = "double" if feature_df[col].dtype in ['float64', 'float32', 'float'] else "bigint"
                new_features.append(Feature(col, dtype))
        if new_features:
            print(f"Schema Migration: Appending new features to Hopsworks Feature Group: {[f.name for f in new_features]}...")
            try:
                aqi_fg.append_features(new_features)
            except Exception as e:
                print(f"Note on append_features: {e}")

        print("Persisting feature dataset to Hopsworks Feature Group...")
        aqi_fg.insert(feature_df, storage="online", wait=True)
        print("Successfully synced features to Hopsworks Feature Store.")
    except Exception as err:
        print(f"Hopsworks sync note ({err}). Local warm cache preserved.")






if __name__ == "__main__":
    run_feature_pipeline()
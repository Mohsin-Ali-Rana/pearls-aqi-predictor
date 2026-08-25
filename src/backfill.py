import argparse
from datetime import datetime, timedelta, timezone
import pandas as pd
import hopsworks

try:
    from fetch_raw_data import fetch_historical_aqi
    from feature_pipeline import generate_features
    from config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )
except ImportError:
    from src.fetch_raw_data import fetch_historical_aqi
    from src.feature_pipeline import generate_features
    from src.config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )

def run_backfill(days: int = 730, start_date: str = None, end_date: str = None):
    """
    Historical feature range backfill script.
    Populates Hopsworks Feature Store with 2-year (730 days) historical feature data.
    """
    today_utc = datetime.now(timezone.utc)
    if not end_date:
        end_date = today_utc.strftime("%Y-%m-%d")
    if not start_date:
        start_date = (today_utc - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"=== Starting Historical Backfill Pipeline ===")
    print(f"Target Period: {start_date} to {end_date} ({days} days)")
    print(f"Location Coordinates: ({LOCATION_LATITUDE}, {LOCATION_LONGITUDE})")

    print("\n1. Fetching historical raw weather & pollution records...")
    raw_df = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, start_date, end_date)
    if raw_df.empty:
        raise ValueError(f"No raw historical data returned from API for range {start_date} to {end_date}")

    print(f"Fetched {len(raw_df)} raw hourly records.")

    print("\n2. Engineering features & time-series lag variables...")
    feature_df = generate_features(raw_df)
    print(f"Engineered feature set contains {len(feature_df)} rows and {len(feature_df.columns)} columns.")

    print("\n3. Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()

    print("\n4. Getting/Creating Feature Group 'aqi_hourly_features' v2...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_hourly_features",
        version=2,
        primary_key=["time"],
        event_time="time",
        online_enabled=True,
        description="Hourly engineered air quality, meteorological and cyclical time features (Historical Backfill)"
    )

    print("\n5. Backfilling feature dataset to Hopsworks Feature Store...")
    aqi_fg.insert(feature_df, storage="online", wait=True)
    print(f"✅ Historical Backfill Complete! Successfully populated {len(feature_df)} rows to Hopsworks Feature Store.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PEARLS AQI Predictor - Historical Feature Store Backfill Tool")
    parser.add_argument("--days", type=int, default=730, help="Number of historical days to backfill (default: 730)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    run_backfill(days=args.days, start_date=args.start_date, end_date=args.end_date)

import hopsworks
import pandas as pd
from fetch_raw_data import fetch_historical_aqi
from feature_pipeline import generate_features
from config import LOCATION_LATITUDE, LOCATION_LONGITUDE

def upload_features():
    # 1. Fetch and engineer features
    print("Step 1: Ingesting and engineering features...")
    raw_df = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, "2026-06-01", "2026-07-21")
    feature_df = generate_features(raw_df)
    
    # CRITICAL FIX 1: Ensure 'time' is a true datetime object, not a string
    feature_df["time"] = pd.to_datetime(feature_df["time"])
    
    # Optional: Reset index if 'time' accidentally became the dataframe index
    if "time" not in feature_df.columns and feature_df.index.name == "time":
        feature_df = feature_df.reset_index()

    # 2. Login to Hopsworks Feature Store
    print("\nStep 2: Authenticating with Hopsworks...")
    # Pass your API key directly here or via an environment variable
    # project = hopsworks.login(api_key_value="yC0HPp2g2yuZjpXH.9FmXmXktv80uISKXoBZtImHRILwMNxReE2JhWfWX2oVkBVrBJt4JPV7OQGAbMuDu")
    project = hopsworks.login(
        project="MA",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value="yC0HPp2g2yuZjpXH.9FmXmXktv80uISKXoBZtImHRILwMNxReE2JhWfWX2oVkBVrBJt4JPV7OQGAbMuDu"
    )
    fs = project.get_feature_store()
    
# 3. Create Feature Group with Online Storage Enabled
    print("\nStep 3: Uploading Feature Group to Cloud Feature Store...")
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_hourly_features",
        version=1,
        primary_key=["time"],
        event_time="time",
        online_enabled=True,  # Enables online database storage
        description="Hourly engineered air quality and pollutant features"
    )
    
    # 4. Write directly to the online feature store
    print("Persisting data to Hopsworks...")
    aqi_fg.insert(feature_df, storage="online", wait=True)
        
    print("\n✅ Successfully persisted features to Hopsworks Feature Store!")

if __name__ == "__main__":
    upload_features()

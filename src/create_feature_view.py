import hopsworks
import pandas as pd
from typing import Tuple

def create_feature_view_and_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Connects to Hopsworks, registers the Feature View metadata, and reads our 
    engineered features from the Online Store to generate chronological splits.
    """
    # ----------------------------------------------------
    # 1. Authenticate & Connect to Feature Store
    # ----------------------------------------------------
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project="MA",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value="yC0HPp2g2yuZjpXH.9FmXmXktv80uISKXoBZtImHRILwMNxReE2JhWfWX2oVkBVrBJt4JPV7OQGAbMuDu"  # Replace with your actual key
    )
    fs = project.get_feature_store()
    
    # ----------------------------------------------------
    # 2. Retrieve the Feature Group & Register Feature View
    # ----------------------------------------------------
    print("Fetching 'aqi_hourly_features' feature group...")
    aqi_fg = fs.get_feature_group(name="aqi_hourly_features", version=1)
    
    # Select all engineered features
    ds_query = aqi_fg.select_all()
    
    print("Registering Feature View metadata: 'aqi_hourly_feature_view'...")
    feature_view = fs.get_or_create_feature_view(
        name="aqi_hourly_feature_view",
        version=1,
        description="Hourly feature view for PM2.5 time-series regression forecasting",
        labels=["pm2_5"],  # Designate PM2.5 as our target (y)
        query=ds_query
    )
    
    # ----------------------------------------------------
    # 3. Read Data Directly from the Online Feature Store
    # ----------------------------------------------------
    # Bypasses offline HDFS entirely and pulls our 1,200 rows from online RonDB
    print("Reading 1,200 feature rows from Online Storage...")
    df = aqi_fg.read(online=True)
    
    # Sort chronologically by timestamp to prevent temporal data leakage
    df = df.sort_values("time").reset_index(drop=True)
    
    # ----------------------------------------------------
    # 4. Perform Chronological 80/20 Train/Test Split
    # ----------------------------------------------------
    print("Generating chronological train/test dataset splits (80/20)...")
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Separate input predictors (X) from the ground-truth label (y)
    target_col = "pm2_5"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[[target_col]]
    
    X_test = test_df[feature_cols]
    y_test = test_df[[target_col]]
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    print("Starting Phase 8: Feature View & Dataset Generation Pipeline...")
    
    X_train, X_test, y_train, y_test = create_feature_view_and_splits()
    
    print("\n--- Dataset Split Summary ---")
    print(f"Training Features Shape (X_train): {X_train.shape}")
    print(f"Training Targets Shape  (y_train): {y_train.shape}")
    print(f"Testing Features Shape  (X_test):  {X_test.shape}")
    print(f"Testing Targets Shape   (y_test):  {y_test.shape}")
    
    print("\nTraining Features Preview (First 3 Rows):")
    preview_cols = [col for col in ['time', 'pm10', 'pm2_5_lag_1h', 'pm2_5_rolling_24h_mean', 'aqi_change_rate'] if col in X_train.columns]
    print(X_train[preview_cols].head(3))
    
    print("\nTraining Target Preview (First 3 Rows):")
    print(y_train.head(3))
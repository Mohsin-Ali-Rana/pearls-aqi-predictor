import os
import hopsworks
import pandas as pd
import joblib

def run_inference():
    """
    Connects to Hopsworks, downloads the best registered model,
    pulls the latest features, and generates real-time AQI predictions
    directly in memory per project architecture guidelines.
    """
    # ----------------------------------------------------
    # 1. Connect & Retrieve Model from Hopsworks Registry
    # ----------------------------------------------------
    print("Connecting to Hopsworks Feature Store & Registry...")
    project = hopsworks.login(
        project="MA",
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value="yC0HPp2g2yuZjpXH.9FmXmXktv80uISKXoBZtImHRILwMNxReE2JhWfWX2oVkBVrBJt4JPV7OQGAbMuDu"  # Put your exact Hopsworks API key here
    )

    mr = project.get_model_registry()
    
    print("Fetching latest version of 'aqi_pm25_predictor' model...")
    model_meta = mr.get_model("aqi_pm25_predictor", version=1)
    model_dir = model_meta.download()
    
    # Identify model type based on downloaded files and load into memory
    if os.path.exists(os.path.join(model_dir, "model.json")):
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(os.path.join(model_dir, "model.json"))
        print("Successfully loaded XGBoost production model.")
    elif os.path.exists(os.path.join(model_dir, "model.txt")):
        import lightgbm as lgb
        model = lgb.Booster(model_file=os.path.join(model_dir, "model.txt"))
        print("Successfully loaded LightGBM production model.")
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        print("Successfully loaded RandomForest production model.")

    # ----------------------------------------------------
    # 2. Pull Fresh Feature Data for Prediction Window
    # ----------------------------------------------------
    fs = project.get_feature_store()
    
    print("Loading feature data locally for immediate inference...")
    fg = fs.get_feature_group("aqi_hourly_features", version=1)
    
    # Read the latest records using standard pandas read parameters
    batch_data = fg.read(online=True)
    batch_data = batch_data.sort_values("time").reset_index(drop=True)

    # Drop non-feature metadata columns if present
    target_col = "pm2_5"
    drop_cols = [target_col, "time"] if "time" in batch_data.columns else [target_col]
    feature_cols = [col for col in batch_data.columns if col not in drop_cols]
    
    X_inference = batch_data[feature_cols].tail(3) # Generating forecast window for the latest 3 timestamps

    # ----------------------------------------------------
    # 3. Generate Forecast Predictions
    # ----------------------------------------------------
    print("\n--- Generating Real-Time Forecasts ---")
    predictions = model.predict(X_inference)
    
    for idx, pred in enumerate(predictions):
        print(f" └─ Forecast Horizon +{idx+1}h -> Predicted PM2.5: {pred:.2f} µg/m³")

    print("\n✅ Inference pipeline executed successfully directly from Hopsworks assets!")
    return predictions

if __name__ == "__main__":
    print("Starting Automated Inference & Prediction Pipeline...")
    run_inference()
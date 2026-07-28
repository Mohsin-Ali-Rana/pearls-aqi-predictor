import os
import hopsworks
import pandas as pd
import joblib
from datetime import datetime

def run_inference():
    """
    Connects to Hopsworks, downloads the best registered model,
    pulls the latest features, and generates real-time AQI predictions.
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
    # Pull the latest batch using the feature group's internal client read with low timeout
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

    # ----------------------------------------------------
    # 4. Export Predictions for Dashboard / Application Layer
    # ----------------------------------------------------
    output_dir = "src"
    os.makedirs(output_dir, exist_ok=True)
    
    output_df = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now(), periods=3, freq='h'),
        'predicted_pm2_5': predictions
    })
    
    output_path = os.path.join(output_dir, "latest_predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"✅ Successfully saved predictions to {output_path}!")

if __name__ == "__main__":
    print("Starting Phase 11: Automated Inference & Prediction Pipeline...")
    run_inference()
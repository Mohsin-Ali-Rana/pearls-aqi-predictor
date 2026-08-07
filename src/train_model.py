import os
import hopsworks
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT

def train_evaluate_and_register_best_model():
    """
    Trains multiple regression models, evaluates performance on chronological splits,
    selects the winner based on lowest RMSE, and registers it to Hopsworks.
    """
    # ----------------------------------------------------
    # 1. Connect & Retrieve Data from Feature Store
    # ----------------------------------------------------
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host=HOPSWORKS_HOST,
        port=HOPSWORKS_PORT,
        api_key_value=HOPSWORKS_API_KEY
    )
    fs = project.get_feature_store()
    
    print("Fetching Feature View: 'aqi_hourly_feature_view'...")
    feature_view = fs.get_feature_view(name="aqi_hourly_feature_view", version=1)
    
    print("Reading dataset from Online Storage...")
    aqi_fg = fs.get_feature_group(name="aqi_hourly_features", version=1)
    df = aqi_fg.read(online=True)
    df = df.sort_values("time").reset_index(drop=True)
    
    # Chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    target_col = "pm2_5"
    drop_cols = [target_col, "time"] if "time" in df.columns else [target_col]
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # ----------------------------------------------------
    # 2. Define Model Candidates
    # ----------------------------------------------------
    candidates = {
        "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    # ----------------------------------------------------
    # 3. Train & Evaluate All Models
    # ----------------------------------------------------
    print("\n--- Starting Model Tournament ---")
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        # Fixed for newer scikit-learn versions: calculate MSE then take the square root
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)   # 1. Calculate MAE
        r2 = r2_score(y_test, predictions)
        
        results[name] = {"rmse": float(rmse),"mae": float(mae), "r2": float(r2)}
        trained_models[name] = model
        print(f"  └─ {name} -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        
    # ----------------------------------------------------
    # 4. Select the Winner (Lowest RMSE)
    # ----------------------------------------------------
    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 WINNER: {best_model_name} (Lowest RMSE: {best_metrics['rmse']:.4f})")
    
    # ----------------------------------------------------
    # 5. Save & Register Winner in Hopsworks
    # ----------------------------------------------------
    model_dir = "aqi_best_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model artifact locally based on framework type
    if best_model_name == "XGBoost":
        best_model.save_model(os.path.join(model_dir, "model.json"))
    elif best_model_name == "LightGBM":
        best_model.booster_.save_model(os.path.join(model_dir, "model.txt"))
    else:  # RandomForest
        import joblib
        joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
        
    print("Uploading winning model to Hopsworks Model Registry...")
    mr = project.get_model_registry()
    
    hopsworks_model = mr.python.create_model(
        name="aqi_pm25_predictor",
        metrics=best_metrics,
        description=f"Best model ({best_model_name}) trained for hourly PM2.5 forecasting"
    )
    
    hopsworks_model.save(model_dir)
    print(f"✅ Successfully registered winning model ({best_model_name}) to Hopsworks Model Registry!")

if __name__ == "__main__":
    print("Starting Phase 9: Multi-Model Tournament & Registry Pipeline...")
    train_evaluate_and_register_best_model()
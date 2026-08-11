import os
import hopsworks
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
except ImportError:
    from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT

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
    
    # Helper function for horizon metrics calculation
    def get_horizon_metrics(y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        m_mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(np.sqrt(m_mse)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))
        }

    # ----------------------------------------------------
    # 3. Train & Evaluate All Models across Horizons
    # ----------------------------------------------------
    print("\n--- Starting Model Tournament & Horizon Evaluation ---")
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        overall_mse = mean_squared_error(y_test, predictions)
        overall_rmse = float(np.sqrt(overall_mse))
        overall_mae = float(mean_absolute_error(y_test, predictions))
        overall_r2 = float(r2_score(y_test, predictions))

        # Horizon metric slicing (Day 1: 1-24, Day 2: 25-48, Day 3: 49-72, Overall 72h)
        y_test_vals = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        d1_m = get_horizon_metrics(y_test_vals[0:min(24, len(y_test_vals))], predictions[0:min(24, len(predictions))])
        d2_m = get_horizon_metrics(y_test_vals[24:min(48, len(y_test_vals))], predictions[24:min(48, len(predictions))])
        d3_m = get_horizon_metrics(y_test_vals[48:min(72, len(y_test_vals))], predictions[48:min(72, len(predictions))])
        h72_m = get_horizon_metrics(y_test_vals[0:min(72, len(y_test_vals))], predictions[0:min(72, len(predictions))])
        
        results[name] = {
            "rmse": overall_rmse,
            "mae": overall_mae,
            "r2": overall_r2,
            "day1_rmse": d1_m["rmse"],
            "day1_mae": d1_m["mae"],
            "day1_r2": d1_m["r2"],
            "day2_rmse": d2_m["rmse"],
            "day2_mae": d2_m["mae"],
            "day2_r2": d2_m["r2"],
            "day3_rmse": d3_m["rmse"],
            "day3_mae": d3_m["mae"],
            "day3_r2": d3_m["r2"],
            "overall_72h_rmse": h72_m["rmse"],
            "overall_72h_mae": h72_m["mae"],
            "overall_72h_r2": h72_m["r2"],
        }
        trained_models[name] = model
        print(f"  └─ {name} Overall -> RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} | R2: {overall_r2:.4f}")
        print(f"     Day 1 (Hours 1–24)  -> RMSE: {d1_m['rmse']:.4f} | MAE: {d1_m['mae']:.4f} | R2: {d1_m['r2']:.4f}")
        print(f"     Day 2 (Hours 25–48) -> RMSE: {d2_m['rmse']:.4f} | MAE: {d2_m['mae']:.4f} | R2: {d2_m['r2']:.4f}")
        print(f"     Day 3 (Hours 49–72) -> RMSE: {d3_m['rmse']:.4f} | MAE: {d3_m['mae']:.4f} | R2: {d3_m['r2']:.4f}")
        print(f"     Overall 72-Hour     -> RMSE: {h72_m['rmse']:.4f} | MAE: {h72_m['mae']:.4f} | R2: {h72_m['r2']:.4f}")
        
    # ----------------------------------------------------
    # 4. Select Tournament Winner (Lowest Overall RMSE)
    # ----------------------------------------------------
    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 TOURNAMENT WINNER: {best_model_name} (Lowest RMSE: {best_metrics['rmse']:.4f} | MAE: {best_metrics['mae']:.4f})")
    
    # ----------------------------------------------------
    # 5. Model Promotion Gate & Registry Serving
    # ----------------------------------------------------
    print("\n--- Automated Model Promotion Gate ---")
    mr = project.get_model_registry()
    
    existing_models = mr.get_models("aqi_pm25_predictor")
    champion_metrics = None
    champion_version = None
    
    if existing_models:
        champion_model = max(existing_models, key=lambda m: int(m.version))
        champion_version = champion_model.version
        champion_metrics = champion_model.training_metrics or {}
        print(f"Active Champion Model: Version {champion_version}")
        champ_rmse_str = f"{champion_metrics.get('rmse'):.4f}" if isinstance(champion_metrics.get('rmse'), (int, float)) else "N/A"
        champ_mae_str = f"{champion_metrics.get('mae'):.4f}" if isinstance(champion_metrics.get('mae'), (int, float)) else "N/A"
        print(f"  └─ Champion Metrics -> RMSE: {champ_rmse_str} | MAE: {champ_mae_str}")
    else:
        print("No existing champion model found in Model Registry. Initial model will be promoted automatically.")

    promote_model = False
    gate_reason = ""
    
    if champion_metrics is None or "rmse" not in champion_metrics:
        promote_model = True
        gate_reason = "Initial champion model registration."
    else:
        champion_rmse = float(champion_metrics["rmse"])
        champion_mae = float(champion_metrics.get("mae", champion_rmse))
        
        new_rmse = best_metrics["rmse"]
        new_mae = best_metrics["mae"]
        
        primary_pass = new_rmse < champion_rmse
        secondary_pass = new_mae <= (champion_mae * 1.02)
        
        if primary_pass and secondary_pass:
            promote_model = True
            gate_reason = f"Primary Gate PASSED (New RMSE {new_rmse:.4f} < Champion RMSE {champion_rmse:.4f}) AND Secondary Guard PASSED (New MAE {new_mae:.4f} <= Max MAE {champion_mae * 1.02:.4f})."
        else:
            promote_model = False
            reasons = []
            if not primary_pass:
                reasons.append(f"Primary Gate Failed: New RMSE ({new_rmse:.4f}) >= Champion RMSE ({champion_rmse:.4f})")
            if not secondary_pass:
                reasons.append(f"Secondary Safety Guard Failed: New MAE ({new_mae:.4f}) > Allowed Champion Threshold ({champion_mae * 1.02:.4f})")
            gate_reason = " | ".join(reasons)

    if promote_model:
        print(f"✅ PROMOTION APPROVED: {gate_reason}")
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
            
        print("Uploading promoted champion model to Hopsworks Model Registry...")
        hopsworks_model = mr.python.create_model(
            name="aqi_pm25_predictor",
            metrics=best_metrics,
            description=f"Promoted champion model ({best_model_name}) trained for hourly PM2.5 forecasting"
        )
        hopsworks_model.save(model_dir)
        print(f"✅ Successfully registered winning champion model ({best_model_name}) to Hopsworks Model Registry!")
    else:
        print(f"🛑 PROMOTION GATE REJECTED: {gate_reason}")
        print(f"Logged run results for winner ({best_model_name}), retaining current champion (Version {champion_version}) for live serving.")

if __name__ == "__main__":
    print("Starting Phase 9: Multi-Model Tournament & Registry Pipeline...")
    train_evaluate_and_register_best_model()
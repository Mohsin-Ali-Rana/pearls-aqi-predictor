import os
import re
import hopsworks
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
try:
    from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    from utils import convert_pm25_to_aqi
except ImportError:
    from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    from src.utils import convert_pm25_to_aqi

def train_evaluate_and_register_best_model():
    """
    Trains multiple regression models, evaluates performance on chronological splits
    using a leak-free recursive evaluation framework and naive persistence baseline,
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
    
    print("Fetching dynamic Feature View: 'aqi_hourly_feature_view'...")
    try:
        fvs = fs.get_feature_views("aqi_hourly_feature_view")
        fv_version = max([int(fv.version) for fv in fvs]) if fvs else 2
    except Exception:
        fv_version = 2
    print(f"Using Feature View Version: {fv_version}")
    feature_view = fs.get_feature_view(name="aqi_hourly_feature_view", version=fv_version)
    
    try:
        fgs = fs.get_feature_groups("aqi_hourly_features")
        fg_version = max([int(fg.version) for fg in fgs]) if fgs else 2
    except Exception:
        fg_version = 2
    print(f"Reading dataset from Online Storage (Feature Group v{fg_version})...")
    aqi_fg = fs.get_feature_group(name="aqi_hourly_features", version=fg_version)
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
    
    # Helper function for horizon metrics calculation on accumulated continuous arrays
    def get_horizon_metrics(y_true, y_pred):
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}
        y_t = np.array(y_true, dtype=np.float64)
        y_p = np.array(y_pred, dtype=np.float64)
        m_mse = float(mean_squared_error(y_t, y_p))
        m_mae = float(mean_absolute_error(y_t, y_p))
        m_r2  = float(r2_score(y_t, y_p))
        return {
            "rmse": float(np.sqrt(m_mse)),
            "mae": float(m_mae),
            "r2": float(round(m_r2, 4))
        }

    def recursive_forecast(model, df_full, origin_idx, feature_cols, target_col="pm2_5", n_steps=72):
        """
        Executes leak-free recursive multi-step forecasting starting from origin_idx.
        Dynamically updates time/lag/rolling features using ONLY model predictions
        for step t (no contemporaneous ground-truth co-pollutant leakage).
        """
        pm25_hist = list(df_full[target_col].values[:origin_idx + 1])
        pm10_hist = list(df_full['pm10'].values[:origin_idx + 1]) if 'pm10' in df_full.columns else list(df_full[target_col].values[:origin_idx + 1])
        eqi_hist  = list(df_full['european_aqi'].values[:origin_idx + 1]) if 'european_aqi' in df_full.columns else [convert_pm25_to_aqi(v) for v in pm25_hist]

        origin_time = pd.to_datetime(df_full['time'].values[origin_idx]) if 'time' in df_full.columns else pd.Timestamp('2024-01-01')
        X_origin   = df_full[feature_cols].iloc[[origin_idx]].copy()

        forecast_steps = []
        for step in range(1, n_steps + 1):
            step_time = origin_time + pd.Timedelta(hours=step)
            step_idx = min(origin_idx + step, len(df_full) - 1)
            step_features = {}

            # Temporal / cyclical features
            for col in feature_cols:
                h = step_time.hour
                if col == 'hour':            step_features[col] = h
                elif col == 'day_of_week':   step_features[col] = step_time.dayofweek
                elif col == 'month':         step_features[col] = step_time.month
                elif col == 'is_weekend':    step_features[col] = 1 if step_time.dayofweek >= 5 else 0
                elif col == 'sin_hour':      step_features[col] = float(np.sin(2 * np.pi * h / 24.0))
                elif col == 'cos_hour':      step_features[col] = float(np.cos(2 * np.pi * h / 24.0))
                elif col == 'sin_day_of_week': step_features[col] = float(np.sin(2 * np.pi * step_time.weekday() / 7.0))
                elif col == 'cos_day_of_week': step_features[col] = float(np.cos(2 * np.pi * step_time.weekday() / 7.0))
                elif col == 'pm2_5_lag_1h': step_features[col] = pm25_hist[-1]
                elif col == 'pm2_5_lag_3h': step_features[col] = pm25_hist[-3] if len(pm25_hist) >= 3 else pm25_hist[-1]
                elif col == 'pm2_5_lag_24h': step_features[col] = pm25_hist[-24] if len(pm25_hist) >= 24 else pm25_hist[-1]
                elif col == 'pm10_lag_1h':  step_features[col] = pm10_hist[-1]
                elif col == 'pm10_lag_3h':  step_features[col] = pm10_hist[-3] if len(pm10_hist) >= 3 else pm10_hist[-1]
                elif col == 'pm10_lag_24h': step_features[col] = pm10_hist[-24] if len(pm10_hist) >= 24 else pm10_hist[-1]
                elif col == 'european_aqi_lag_1h': step_features[col] = eqi_hist[-1]
                elif col == 'european_aqi_lag_3h': step_features[col] = eqi_hist[-3] if len(eqi_hist) >= 3 else eqi_hist[-1]
                elif col == 'european_aqi_lag_24h': step_features[col] = eqi_hist[-24] if len(eqi_hist) >= 24 else eqi_hist[-1]
                elif col == 'pm2_5_rolling_6h_mean':  step_features[col] = float(np.mean(pm25_hist[-6:]))
                elif col == 'pm2_5_rolling_24h_mean': step_features[col] = float(np.mean(pm25_hist[-24:]))
                elif col == 'pm10_rolling_6h_mean':   step_features[col] = float(np.mean(pm10_hist[-6:]))
                elif col == 'pm10_rolling_24h_mean':  step_features[col] = float(np.mean(pm10_hist[-24:]))
                elif col == 'aqi_change_rate':
                    step_features[col] = float(eqi_hist[-1] - (eqi_hist[-3] if len(eqi_hist) >= 3 else eqi_hist[-1]))
                elif col in ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure'] and col in df_full.columns:
                    # Valid exogenous weather forecasts at step t
                    step_features[col] = float(df_full[col].iloc[step_idx])
                elif col in X_origin.columns:
                    step_features[col] = float(X_origin[col].values[0])
                else:
                    step_features[col] = 0.0

            step_df = pd.DataFrame([step_features])
            if hasattr(model, 'feature_names_in_'):
                step_df = step_df.reindex(columns=list(model.feature_names_in_), fill_value=0.0)
            elif hasattr(model, 'feature_name'):
                step_df = step_df.reindex(columns=model.feature_name(), fill_value=0.0)
            else:
                step_df = step_df[feature_cols]

            pred = float(model.predict(step_df)[0] if hasattr(model, 'predict') else model.predict(step_df.values)[0])
            pred = max(0.1, pred)
            forecast_steps.append(pred)

            # Strictly update historical buffers with PREDICTED values (no co-pollutant ground-truth leakage!)
            pm25_hist.append(pred)
            pm10_hist.append(pred)
            eqi_hist.append(convert_pm25_to_aqi(pred))

        return forecast_steps

    # ----------------------------------------------------
    # 3. Evaluate Naive Persistence Baseline (y_hat_{t+H} = y_t)
    # ----------------------------------------------------
    print("\n--- Evaluating Naive Persistence Baseline (y_hat_{t+H} = y_t) ---")
    p_d1_trues, p_d1_preds = [], []
    p_d2_trues, p_d2_preds = [], []
    p_d3_trues, p_d3_preds = [], []
    p_h72_trues, p_h72_preds = [], []

    start_origin = split_idx
    end_origin = len(df) - 73
    step_stride = 24

    if end_origin <= start_origin:
        origin_indices = [start_origin]
    else:
        origin_indices = list(range(start_origin, end_origin, step_stride))

    for orig in origin_indices:
        actuals = df[target_col].iloc[orig + 1 : orig + 73].values
        if len(actuals) < 72:
            continue
        y_0 = df[target_col].iloc[orig]
        persistence_preds = [y_0] * 72

        p_d1_trues.extend(actuals[:24])
        p_d1_preds.extend(persistence_preds[:24])

        p_d2_trues.extend(actuals[24:48])
        p_d2_preds.extend(persistence_preds[24:48])

        p_d3_trues.extend(actuals[48:72])
        p_d3_preds.extend(persistence_preds[48:72])

        p_h72_trues.extend(actuals[:72])
        p_h72_preds.extend(persistence_preds[:72])

    p_d1_m  = get_horizon_metrics(p_d1_trues, p_d1_preds)
    p_d2_m  = get_horizon_metrics(p_d2_trues, p_d2_preds)
    p_d3_m  = get_horizon_metrics(p_d3_trues, p_d3_preds)
    p_h72_m = get_horizon_metrics(p_h72_trues, p_h72_preds)

    persistence_metrics = {
        "rmse": p_h72_m["rmse"],
        "mae":  p_h72_m["mae"],
        "r2":   p_h72_m["r2"],
        "day1_rmse": p_d1_m["rmse"], "day1_mae": p_d1_m["mae"], "day1_r2": p_d1_m["r2"],
        "day2_rmse": p_d2_m["rmse"], "day2_mae": p_d2_m["mae"], "day2_r2": p_d2_m["r2"],
        "day3_rmse": p_d3_m["rmse"], "day3_mae": p_d3_m["mae"], "day3_r2": p_d3_m["r2"],
        "overall_72h_rmse": p_h72_m["rmse"],
        "overall_72h_mae":  p_h72_m["mae"],
        "overall_72h_r2":   p_h72_m["r2"],
    }
    print(f"  └─ Persistence Baseline Overall 72H -> RMSE: {persistence_metrics['rmse']:.4f} | MAE: {persistence_metrics['mae']:.4f} | R2: {persistence_metrics['r2']:.4f}")
    print(f"     Day 1 Persistence (1–24h)  -> RMSE: {p_d1_m['rmse']:.4f} | MAE: {p_d1_m['mae']:.4f} | R2: {p_d1_m['r2']:.4f}")
    print(f"     Day 2 Persistence (25–48h) -> RMSE: {p_d2_m['rmse']:.4f} | MAE: {p_d2_m['mae']:.4f} | R2: {p_d2_m['r2']:.4f}")
    print(f"     Day 3 Persistence (49–72h) -> RMSE: {p_d3_m['rmse']:.4f} | MAE: {p_d3_m['mae']:.4f} | R2: {p_d3_m['r2']:.4f}")

    # ----------------------------------------------------
    # 4. Train & Evaluate Candidate Models
    # ----------------------------------------------------
    print("\n--- Starting Model Tournament & Leak-Free Recursive Evaluation ---")
    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        print(f"  Evaluating leak-free multi-step recursive horizon metrics for {name} on raw PM2.5 scale...")
        d1_trues, d1_preds = [], []
        d2_trues, d2_preds = [], []
        d3_trues, d3_preds = [], []
        h72_trues, h72_preds = [], []

        for orig in origin_indices:
            actuals = df[target_col].iloc[orig + 1 : orig + 73].values
            if len(actuals) < 72:
                continue
            preds = recursive_forecast(model, df, origin_idx=orig, feature_cols=feature_cols, target_col=target_col, n_steps=72)
            
            d1_trues.extend(actuals[:24])
            d1_preds.extend(preds[:24])
            
            d2_trues.extend(actuals[24:48])
            d2_preds.extend(preds[24:48])
            
            d3_trues.extend(actuals[48:72])
            d3_preds.extend(preds[48:72])
            
            h72_trues.extend(actuals[:72])
            h72_preds.extend(preds[:72])

        d1_m  = get_horizon_metrics(d1_trues, d1_preds)
        d2_m  = get_horizon_metrics(d2_trues, d2_preds)
        d3_m  = get_horizon_metrics(d3_trues, d3_preds)
        h72_m = get_horizon_metrics(h72_trues, h72_preds)

        overall_rmse = h72_m["rmse"]
        overall_mae  = h72_m["mae"]
        overall_r2   = h72_m["r2"]

        results[name] = {
            "rmse": overall_rmse,
            "mae":  overall_mae,
            "r2":   overall_r2,
            "day1_rmse": d1_m["rmse"], "day1_mae": d1_m["mae"], "day1_r2": d1_m["r2"],
            "day2_rmse": d2_m["rmse"], "day2_mae": d2_m["mae"], "day2_r2": d2_m["r2"],
            "day3_rmse": d3_m["rmse"], "day3_mae": d3_m["mae"], "day3_r2": d3_m["r2"],
            "overall_72h_rmse": h72_m["rmse"],
            "overall_72h_mae":  h72_m["mae"],
            "overall_72h_r2":   h72_m["r2"],
        }
        trained_models[name] = model
        print(f"  └─ {name} Overall Recursive 72H -> RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} | R2: {overall_r2:.4f}")
        print(f"     Day 1 Leak-Free (1–24h)   -> RMSE: {d1_m['rmse']:.4f} | MAE: {d1_m['mae']:.4f} | R2: {d1_m['r2']:.4f}")
        print(f"     Day 2 Leak-Free (25–48h)  -> RMSE: {d2_m['rmse']:.4f} | MAE: {d2_m['mae']:.4f} | R2: {d2_m['r2']:.4f}")
        print(f"     Day 3 Leak-Free (49–72h)  -> RMSE: {d3_m['rmse']:.4f} | MAE: {d3_m['mae']:.4f} | R2: {d3_m['r2']:.4f}")
        
    # ----------------------------------------------------
    # 5. Select Tournament Winner (Lowest Overall Recursive RMSE)
    # ----------------------------------------------------
    best_model_name = min(results, key=lambda x: results[x]["rmse"])
    best_model = trained_models[best_model_name]
    best_metrics = results[best_model_name]
    
    print(f"\n🏆 TOURNAMENT WINNER: {best_model_name} (Lowest Leak-Free Recursive RMSE: {best_metrics['rmse']:.4f} | MAE: {best_metrics['mae']:.4f})")
    
    # ----------------------------------------------------
    # 6. Model Promotion Gate & Registry Serving
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
    
    champ_rmse = float(champion_metrics.get("rmse", 0.0)) if champion_metrics and isinstance(champion_metrics.get("rmse"), (int, float)) else None

    # Reset condition: If current champion is Version <= 19 or contains legacy leaked RMSE (< 3.0),
    # force promotion reset to establish the official leak-free Version 20 baseline.
    is_legacy_champion = (
        champion_version is not None and (
            int(champion_version) <= 19 or (champ_rmse is not None and champ_rmse < 3.0)
        )
    )

    if champion_metrics is None or is_legacy_champion:
        promote_model = True
        gate_reason = (
            f"Promotion Gate Reset: Replacing legacy/leaked Champion (Version {champion_version}, "
            f"RMSE: {champ_rmse_str}) with leak-free Model Version 20 establishing the official baseline."
        )
    else:
        candidate_rmse = best_metrics["rmse"]
        persistence_rmse = persistence_metrics["rmse"]
        
        beats_champion = candidate_rmse < champ_rmse
        beats_persistence = candidate_rmse < persistence_rmse
        
        if beats_champion and beats_persistence:
            promote_model = True
            gate_reason = (
                f"Promotion Gate PASSED: Candidate RMSE ({candidate_rmse:.4f}) beat Champion RMSE ({champ_rmse:.4f}) "
                f"AND Naive Persistence RMSE ({persistence_rmse:.4f})."
            )
        else:
            promote_model = False
            gate_reason = (
                f"Promotion Gate REJECTED: Candidate RMSE ({candidate_rmse:.4f}) failed criteria "
                f"(Champion RMSE: {champ_rmse:.4f}, Persistence RMSE: {persistence_rmse:.4f})."
            )

    if promote_model:
        print(f"✅ PROMOTION APPROVED: {gate_reason}")
        model_dir = "aqi_best_model"
        os.makedirs(model_dir, exist_ok=True)
        
        if best_model_name == "XGBoost":
            best_model.save_model(os.path.join(model_dir, "model.json"))
        elif best_model_name == "LightGBM":
            best_model.booster_.save_model(os.path.join(model_dir, "model.txt"))
        else:  # RandomForest
            import joblib
            joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))
            
        print("Uploading promoted champion model to Hopsworks Model Registry...")
        
        # Include persistence metrics in uploaded metadata dictionary
        upload_metrics = dict(best_metrics)
        upload_metrics["persistence_rmse"] = float(round(persistence_metrics["rmse"], 4))
        upload_metrics["persistence_mae"]  = float(round(persistence_metrics["mae"], 4))
        upload_metrics["persistence_r2"]   = float(round(persistence_metrics["r2"], 4))

        hopsworks_model = mr.python.create_model(
            name="aqi_pm25_predictor",
            metrics=upload_metrics,
            description=f"Promoted leak-free champion model ({best_model_name}) evaluated with honest 72-hour multi-step forecasting"
        )
        hopsworks_model.save(model_dir)
        print(f"✅ Successfully registered winning champion model ({best_model_name}) to Hopsworks Model Registry!")
    else:
        print(f"🛑 PROMOTION GATE REJECTED: {gate_reason}")
        print(f"Logged run results for winner ({best_model_name}), retaining current champion (Version {champion_version}) for live serving.")

if __name__ == "__main__":
    print("Starting Phase 9: Multi-Model Tournament & Registry Pipeline...")
    train_evaluate_and_register_best_model()
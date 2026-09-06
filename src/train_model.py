import os
import json
import re
import hopsworks
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

try:
    from config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    from utils import convert_pm25_to_aqi
except ImportError:
    from src.config import HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    from src.utils import convert_pm25_to_aqi


# Evaluation metrics calculation function (RMSE, MAE, R2 score)
def eval_metrics(y_true, y_pred):
    y_t = np.array(y_true, dtype=np.float64)
    y_p = np.array(y_pred, dtype=np.float64)
    mse = float(mean_squared_error(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    r2  = float(r2_score(y_t, y_p))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "r2": float(round(r2, 4))
    }






# --- Stage: Model Training & Evaluation Pipeline ---
# 3 Direct Multi-Horizon models ko train, evaluate aur Hopsworks registry par upload karne ka function
def train_evaluate_and_register_best_model():
    # Feature Store se data fetch kar rahe hain ya local warm cache load kar rahe hain
    df = None
    project = None
    mr = None
    try:
        print("Connecting to Hopsworks Feature Store...")
        import concurrent.futures
        def _hw_login():
            return hopsworks.login(
                project=HOPSWORKS_PROJECT,
                host=HOPSWORKS_HOST,
                port=HOPSWORKS_PORT,
                api_key_value=HOPSWORKS_API_KEY
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(_hw_login)
            project = fut.result(timeout=15.0)

        fs = project.get_feature_store()
        try:
            fgs = fs.get_feature_groups("aqi_hourly_features")
            fg_version = max([int(fg.version) for fg in fgs]) if fgs else 2
        except Exception:
            fg_version = 2
        print(f"Reading dataset from Online Storage (Feature Group v{fg_version})...")
        aqi_fg = fs.get_feature_group(name="aqi_hourly_features", version=fg_version)
        import concurrent.futures
        def _read_fg():
            try:
                return aqi_fg.read(read_options={"use_hive": False})
            except Exception:
                return aqi_fg.read()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(_read_fg)
            df = fut.result(timeout=20.0)
    except Exception as e:
        print(f"Feature store query note ({e}). Loading dataset from 'data/features.parquet'...")
        if os.path.exists(os.path.join("data", "features.parquet")):
            df = pd.read_parquet(os.path.join("data", "features.parquet"))
        elif os.path.exists(os.path.join("data", "processed_aqi.csv")):
            df = pd.read_csv(os.path.join("data", "processed_aqi.csv"))
        else:
            raise FileNotFoundError("Could not connect to Hopsworks and no local feature dataset exists.")

    df = df.sort_values("time").reset_index(drop=True)

    # 3 Direct Multi-Horizon Target Columns banaye hain (+24h, +48h, +72h)
    print("Constructing 3 Direct Multi-Horizon Target Arrays...")
    df['target_24h'] = df['pm2_5'].shift(-24)
    df['target_48h'] = df['pm2_5'].shift(-48)
    df['target_72h'] = df['pm2_5'].shift(-72)

    df = df.dropna(subset=['target_24h', 'target_48h', 'target_72h']).reset_index(drop=True)
    print(f"Dataset shape after target alignment: {df.shape}")

    target_cols = ['pm2_5', 'target_24h', 'target_48h', 'target_72h']
    drop_cols = target_cols + (['time'] if 'time' in df.columns else [])
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    print(f"Feature set count: {len(feature_cols)} features.")

    # Chronological 70% Train / 15% Validation / 15% Test split (Data leakage roknay ke liye)
    n_total = len(df)
    train_end = int(n_total * 0.70)
    val_end   = int(n_total * 0.85)

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    X_train = train_df[feature_cols]
    X_val   = val_df[feature_cols]
    X_test  = test_df[feature_cols]

    y_24h_train = train_df['target_24h']
    y_24h_val   = val_df['target_24h']
    y_24h_test  = test_df['target_24h']

    y_48h_train = train_df['target_48h']
    y_48h_val   = val_df['target_48h']
    y_48h_test  = test_df['target_48h']

    y_72h_train = train_df['target_72h']
    y_72h_val   = val_df['target_72h']
    y_72h_test  = test_df['target_72h']

    # Naive Persistence Baseline evaluation (y_hat_{t+H} = y_t)
    y_0_test = test_df['pm2_5']
    print("Evaluating Naive Persistence Baseline (y_hat_{t+H} = y_t)...")
    p_d1_m = eval_metrics(y_24h_test, y_0_test)
    p_d2_m = eval_metrics(y_48h_test, y_0_test)
    p_d3_m = eval_metrics(y_72h_test, y_0_test)

    persistence_rmse = (p_d1_m["rmse"] + p_d2_m["rmse"] + p_d3_m["rmse"]) / 3.0
    persistence_mae  = (p_d1_m["mae"]  + p_d2_m["mae"]  + p_d3_m["mae"])  / 3.0
    persistence_r2   = (p_d1_m["r2"]   + p_d2_m["r2"]   + p_d3_m["r2"])   / 3.0

    print(f"  Day 1 Persistence (24h) -> RMSE: {p_d1_m['rmse']:.4f} | MAE: {p_d1_m['mae']:.4f} | R2: {p_d1_m['r2']:.4f}")
    print(f"  Day 2 Persistence (48h) -> RMSE: {p_d2_m['rmse']:.4f} | MAE: {p_d2_m['mae']:.4f} | R2: {p_d2_m['r2']:.4f}")
    print(f"  Day 3 Persistence (72h) -> RMSE: {p_d3_m['rmse']:.4f} | MAE: {p_d3_m['mae']:.4f} | R2: {p_d3_m['r2']:.4f}")
    print(f"  Persistence Overall     -> RMSE: {persistence_rmse:.4f} | MAE: {persistence_mae:.4f} | R2: {persistence_r2:.4f}")

    # Candidate Models pool (Target variance ko balance karne ke liye log(1+y) transformation use ki hai)
    def get_candidate_pool():
        base_models = {
            "XGBoost": xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            "Ridge": make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=1.0))
        }
        return {
            name: TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1)
            for name, m in base_models.items()
        }

    # Day 1 (24h) Tournament (Validation set par best model pick kiya)
    print("Day 1 (24h) Direct Model Tournament...")
    d1_candidates = get_candidate_pool()
    d1_results = {}
    for name, model in d1_candidates.items():
        model.fit(X_train, y_24h_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        d1_results[name] = {
            "model": model,
            "val_metrics": eval_metrics(y_24h_val, val_preds),
            "metrics": eval_metrics(y_24h_test, test_preds)
        }
        print(f"  {name:12s} -> Val RMSE: {d1_results[name]['val_metrics']['rmse']:.4f} | Test RMSE: {d1_results[name]['metrics']['rmse']:.4f}")
    
    winner_d1_name = min(d1_results, key=lambda k: d1_results[k]["val_metrics"]["rmse"])

    # Day 2 (48h) Tournament
    print("Day 2 (48h) Direct Model Tournament...")
    d2_candidates = get_candidate_pool()
    d2_results = {}
    for name, model in d2_candidates.items():
        model.fit(X_train, y_48h_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        d2_results[name] = {
            "model": model,
            "val_metrics": eval_metrics(y_48h_val, val_preds),
            "metrics": eval_metrics(y_48h_test, test_preds)
        }
        print(f"  {name:12s} -> Val RMSE: {d2_results[name]['val_metrics']['rmse']:.4f} | Test RMSE: {d2_results[name]['metrics']['rmse']:.4f}")

    winner_d2_name = min(d2_results, key=lambda k: d2_results[k]["val_metrics"]["rmse"])

    # Day 3 (72h) Tournament
    print("Day 3 (72h) Direct Model Tournament...")
    d3_candidates = get_candidate_pool()
    d3_results = {}
    for name, model in d3_candidates.items():
        model.fit(X_train, y_72h_train)
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        d3_results[name] = {
            "model": model,
            "val_metrics": eval_metrics(y_72h_val, val_preds),
            "metrics": eval_metrics(y_72h_test, test_preds)
        }
        print(f"  {name:12s} -> Val RMSE: {d3_results[name]['val_metrics']['rmse']:.4f} | Test RMSE: {d3_results[name]['metrics']['rmse']:.4f}")

    winner_d3_name = min(d3_results, key=lambda k: d3_results[k]["val_metrics"]["rmse"])

    # Final winning models ko Train + Val dataset par re-fit kar rahe hain
    print("Re-fitting selected tournament winners on Train + Validation dataset...")
    train_val_df = pd.concat([train_df, val_df]).sort_values("time").reset_index(drop=True)
    X_train_val = train_val_df[feature_cols]
    y_24h_tv    = train_val_df['target_24h']
    y_48h_tv    = train_val_df['target_48h']
    y_72h_tv    = train_val_df['target_72h']

    model_day1 = get_candidate_pool()[winner_d1_name]
    model_day1.fit(X_train_val, y_24h_tv)
    d1_test_preds = model_day1.predict(X_test)
    d1_m = eval_metrics(y_24h_test, d1_test_preds)

    model_day2 = get_candidate_pool()[winner_d2_name]
    model_day2.fit(X_train_val, y_48h_tv)
    d2_test_preds = model_day2.predict(X_test)
    d2_m = eval_metrics(y_48h_test, d2_test_preds)

    model_day3 = get_candidate_pool()[winner_d3_name]
    model_day3.fit(X_train_val, y_72h_tv)
    d3_test_preds = model_day3.predict(X_test)
    d3_m = eval_metrics(y_72h_test, d3_test_preds)

    print(f"Day 1 Winner: {winner_d1_name} (Holdout Test RMSE: {d1_m['rmse']:.4f})")
    print(f"Day 2 Winner: {winner_d2_name} (Holdout Test RMSE: {d2_m['rmse']:.4f})")
    print(f"Day 3 Winner: {winner_d3_name} (Holdout Test RMSE: {d3_m['rmse']:.4f})")

    # Overall metrics aur persistence lift calculate kar rahe hain
    overall_rmse = float(np.mean([d1_m['rmse'], d2_m['rmse'], d3_m['rmse']]))
    overall_mae  = float(np.mean([d1_m['mae'], d2_m['mae'], d3_m['mae']]))
    overall_r2   = float(np.mean([d1_m['r2'], d2_m['r2'], d3_m['r2']]))

    d1_lift_rmse_pct = float(round(((p_d1_m["rmse"] - d1_m["rmse"]) / p_d1_m["rmse"]) * 100.0, 2))
    d2_lift_rmse_pct = float(round(((p_d2_m["rmse"] - d2_m["rmse"]) / p_d2_m["rmse"]) * 100.0, 2))
    d3_lift_rmse_pct = float(round(((p_d3_m["rmse"] - d3_m["rmse"]) / p_d3_m["rmse"]) * 100.0, 2))
    overall_lift_rmse_pct = float(round(((persistence_rmse - overall_rmse) / persistence_rmse) * 100.0, 2))

    print("Final Direct Multi-Horizon Summary:")
    print(f"  Day 1 (24h) [{winner_d1_name:10s}] -> RMSE: {d1_m['rmse']:.4f} | MAE: {d1_m['mae']:.4f} | R2: {d1_m['r2']:.4f} | Persistence Lift: +{d1_lift_rmse_pct:.2f}%")
    print(f"  Day 2 (48h) [{winner_d2_name:10s}] -> RMSE: {d2_m['rmse']:.4f} | MAE: {d2_m['mae']:.4f} | R2: {d2_m['r2']:.4f} | Persistence Lift: +{d2_lift_rmse_pct:.2f}%")
    print(f"  Day 3 (72h) [{winner_d3_name:10s}] -> RMSE: {d3_m['rmse']:.4f} | MAE: {d3_m['mae']:.4f} | R2: {d3_m['r2']:.4f} | Persistence Lift: +{d3_lift_rmse_pct:.2f}%")
    print(f"  Overall 72H Direct Average -> RMSE: {overall_rmse:.4f} | MAE: {overall_mae:.4f} | R2: {overall_r2:.4f} | Overall Lift: +{overall_lift_rmse_pct:.2f}%")

    h24_dict = {name: {**res["metrics"], "winner": (name == winner_d1_name)} for name, res in d1_results.items()}
    h24_dict[winner_d1_name].update({**d1_m, "winner": True})

    h48_dict = {name: {**res["metrics"], "winner": (name == winner_d2_name)} for name, res in d2_results.items()}
    h48_dict[winner_d2_name].update({**d2_m, "winner": True})

    h72_dict = {name: {**res["metrics"], "winner": (name == winner_d3_name)} for name, res in d3_results.items()}
    h72_dict[winner_d3_name].update({**d3_m, "winner": True})

    tournament_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "persistence_baseline": {
            "day1": p_d1_m,
            "day2": p_d2_m,
            "day3": p_d3_m
        },
        "horizons": {
            "24h": h24_dict,
            "48h": h48_dict,
            "72h": h72_dict
        },
        "lifts": {
            "day1_lift_rmse_pct": d1_lift_rmse_pct,
            "day2_lift_rmse_pct": d2_lift_rmse_pct,
            "day3_lift_rmse_pct": d3_lift_rmse_pct,
            "overall_lift_rmse_pct": overall_lift_rmse_pct
        }
    }
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "tournament_summary.json"), "w") as f:
        json.dump(tournament_summary, f, indent=2)
    print("Exported tournament matrix summary to 'data/tournament_summary.json'.")

    best_metrics = {
        "rmse": overall_rmse,
        "mae":  overall_mae,
        "r2":   overall_r2,
        "day1_rmse": float(d1_m["rmse"]), "day1_mae": float(d1_m["mae"]), "day1_r2": float(d1_m["r2"]),
        "day2_rmse": float(d2_m["rmse"]), "day2_mae": float(d2_m["mae"]), "day2_r2": float(d2_m["r2"]),
        "day3_rmse": float(d3_m["rmse"]), "day3_mae": float(d3_m["mae"]), "day3_r2": float(d3_m["r2"]),
        "day1_lift_rmse_pct": d1_lift_rmse_pct,
        "day2_lift_rmse_pct": d2_lift_rmse_pct,
        "day3_lift_rmse_pct": d3_lift_rmse_pct,
        "overall_lift_rmse_pct": overall_lift_rmse_pct,
        "overall_72h_rmse": overall_rmse,
        "overall_72h_mae":  overall_mae,
        "overall_72h_r2":   overall_r2,
        "persistence_rmse": float(persistence_rmse),
        "persistence_mae":  float(persistence_mae),
        "persistence_r2":   float(persistence_r2)
    }

    # SHAP global feature importance calculate kar rahe hain
    print("Computing SHAP Global Feature Importance...")
    def compute_shap_feature_importance(model, X_df, feature_cols):
        raw_m = getattr(model, "regressor_", model)
        if hasattr(raw_m, "named_steps"):
            raw_m = raw_m.named_steps.get("ridge", raw_m)
        try:
            import shap
            explainer = shap.TreeExplainer(raw_m)
            shap_values = explainer.shap_values(X_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(feature_cols, mean_abs_shap.tolist()))
            sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return [{"feature": feat, "importance": round(float(val), 4)} for feat, val in sorted_imp]
        except Exception as e:
            print(f"SHAP TreeExplainer note: {e}. Using model feature_importances_ fallback.")
            if hasattr(raw_m, "feature_importances_"):
                imp = raw_m.feature_importances_
                sorted_imp = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
                return [{"feature": feat, "importance": round(float(val), 4)} for feat, val in sorted_imp]
            return []

    shap_importance_24h = compute_shap_feature_importance(model_day1, X_train, feature_cols)
    print(f"Top 5 SHAP Features (Day 1 Model): {shap_importance_24h[:5]}")

    # Model promotion gate (Active production model RMSE se compare kar rahe hain)
    print("Automated Model Promotion Gate check...")
    mr = project.get_model_registry() if project is not None else None
    
    model_dir = "aqi_best_model"
    os.makedirs(model_dir, exist_ok=True)
    
    reg_name = "aqi_pm25_predictor"
    candidate_rmse = overall_rmse
    should_promote = True
    
    local_pkl = os.path.join(model_dir, "model.pkl")
    if os.path.exists(local_pkl):
        try:
            old_b = joblib.load(local_pkl)
            old_v = int(old_b.get("version", 1))
            target_version = old_v + 1
        except Exception:
            target_version = int(datetime.now().strftime('%m%d%H%M'))
    else:
        target_version = 1

    if mr is not None:
        try:
            existing_models = mr.get_models(reg_name)
            if existing_models:
                latest_model = max(existing_models, key=lambda m: int(getattr(m, "version", 0)))
                latest_ver_num = int(getattr(latest_model, "version", 0))
                target_version = latest_ver_num + 1
                metrics_dict = getattr(latest_model, "metrics", None) or getattr(latest_model, "training_metrics", {}) or {}
                active_rmse_val = metrics_dict.get("rmse")
                if active_rmse_val is not None:
                    active_rmse = float(active_rmse_val)
                    if candidate_rmse >= active_rmse:
                        should_promote = False
                        target_version = latest_ver_num
                        print(f"PROMOTION GATE REJECTED: Candidate model (RMSE: {candidate_rmse:.4f}) did not beat active model (RMSE: {active_rmse:.4f}). Active version v{latest_ver_num} retained.")
                    else:
                        print(f"PROMOTION GATE PASSED: New model beats active production model (Candidate RMSE: {candidate_rmse:.4f} < Active RMSE: {active_rmse:.4f}). Promoting version v{target_version}.")
                else:
                    print(f"PROMOTION GATE PASSED: Existing model version v{latest_ver_num} exists but lacks recorded RMSE metric. Promoting version v{target_version} (Candidate RMSE: {candidate_rmse:.4f}).")
            else:
                print(f"PROMOTION GATE PASSED: No previous model versions exist in registry. Promoting initial version v1 (Candidate RMSE: {candidate_rmse:.4f}).")
        except Exception as e:
            print(f"PROMOTION GATE PASSED: Registry query note ({e}). Promoting candidate version (Candidate RMSE: {candidate_rmse:.4f}).")

    multi_model_bundle = {
        "model_24h": model_day1,
        "model_48h": model_day2,
        "model_72h": model_day3,
        "feature_cols": feature_cols,
        "day1_winner": winner_d1_name,
        "day2_winner": winner_d2_name,
        "day3_winner": winner_d3_name,
        "shap_importance_24h": shap_importance_24h,
        "version": target_version,
        "name": reg_name,
        "training_metrics": best_metrics
    }

    if should_promote:
        pkl_target_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(multi_model_bundle, pkl_target_path)
        print(f"Promoted champion model: Saved bundle to '{pkl_target_path}' (v{target_version}).")

        if mr is not None:
            print(f"Uploading promoted bundle to Hopsworks Model Registry ('{reg_name}')...")
            try:
                hopsworks_model = mr.python.create_model(
                    name=reg_name,
                    metrics=best_metrics,
                    description=f"Promoted 3 Direct Models Architecture (24h={winner_d1_name}, 48h={winner_d2_name}, 72h={winner_d3_name})"
                )
                hopsworks_model.save(model_dir)
                print(f"Successfully registered winning Direct Model bundle to Hopsworks Model Registry ('{reg_name}').")
            except Exception as e:
                print(f"Model registry upload note ({e}). Local artifact successfully ready.")
    else:
        print(f"Model rejected (candidate RMSE >= active RMSE). Preserving existing champion artifact.")
        rejected_target_path = os.path.join("data", "last_rejected_candidate.pkl")
        try:
            joblib.dump(multi_model_bundle, rejected_target_path)
            print(f"Saved rejected candidate artifact for audit to '{rejected_target_path}'.")
        except Exception as e:
            print(f"Note: Could not save rejected candidate artifact ({e}).")






if __name__ == "__main__":
    print("Starting Direct Multi-Horizon 3-Model Tournament Pipeline...")
    train_evaluate_and_register_best_model()
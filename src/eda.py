import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import hopsworks

try:
    from config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )
except ImportError:
    from src.config import (
        LOCATION_LATITUDE, LOCATION_LONGITUDE,
        HOPSWORKS_API_KEY, HOPSWORKS_PROJECT, HOPSWORKS_HOST, HOPSWORKS_PORT
    )


# --- Stage: Exploratory Data Analysis (EDA) ---
# Historical air quality aur weather features ki statistical summary aur correlation analysis running function
def run_eda(save_json: bool = True) -> dict:
    print("Starting Exploratory Data Analysis (EDA)...")
    df = None
    data_source = "Hopsworks Feature Group v2 (aqi_hourly_features)"

    # Hopsworks Feature Store se dataset load kar rahe hain ya local feature store cache use kar rahe hain
    try:
        print("Attempting to load EDA dataset from Hopsworks Feature Store...")
        project = hopsworks.login(
            project=HOPSWORKS_PROJECT,
            host=HOPSWORKS_HOST,
            port=HOPSWORKS_PORT,
            api_key_value=HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        aqi_fg = fs.get_feature_group("aqi_hourly_features", version=2)
        df = aqi_fg.read(read_options={"use_hive": False})
        if df is not None and not df.empty:
            data_source = "Hopsworks Feature Group v2 (aqi_hourly_features)"
            print(f"Successfully retrieved EDA dataset from Hopsworks ({len(df)} records).")
    except Exception as err:
        print(f"Hopsworks query note ({err}). Using local engineered feature store fallback for EDA.")
        df = None

    if df is None or df.empty:
        if os.path.exists(os.path.join("data", "features.parquet")):
            df = pd.read_parquet(os.path.join("data", "features.parquet"))
            data_source = "Local Engineered Feature Store (features.parquet)"
        elif os.path.exists(os.path.join("data", "processed_aqi.csv")):
            df = pd.read_csv(os.path.join("data", "processed_aqi.csv"))
            data_source = "Local Engineered Feature Store (processed_aqi.csv)"
        else:
            raise ValueError("Failed to fetch historical feature dataset for EDA.")

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    # Statistical summary metrics calculate kar rahe hain (mean, std, min, 50%, max)
    numeric_cols = [c for c in df.columns if c not in ['time', 'hour', 'day_of_week']]
    stats_df = df[numeric_cols].describe().T.round(2)
    stats_summary = stats_df.to_dict(orient="index")

    # Target variable (PM2.5) ke saath feature correlation matrix
    corr_matrix = df[numeric_cols].corr().round(4)
    pm25_correlations = corr_matrix['pm2_5'].sort_values(ascending=False).to_dict() if 'pm2_5' in corr_matrix.columns else {}
    full_corr_matrix_dict = corr_matrix.to_dict()

    # 24-hour diurnal cycle profile (Hourly averages of PM2.5, EAQI, temperature, wind speed)
    hourly_avg = df.groupby('hour')[['pm2_5', 'european_aqi', 'temperature_2m', 'wind_speed_10m']].mean().round(2)
    hourly_profile = [
        {
            "hour": f"{h:02d}:00",
            "pm25": float(hourly_avg.loc[h, 'pm2_5']) if ('pm2_5' in hourly_avg.columns and h in hourly_avg.index) else 0.0,
            "aqi": float(hourly_avg.loc[h, 'european_aqi']) if ('european_aqi' in hourly_avg.columns and h in hourly_avg.index) else 0.0,
            "temp": float(hourly_avg.loc[h, 'temperature_2m']) if ('temperature_2m' in hourly_avg.columns and h in hourly_avg.index) else 0.0,
            "wind": float(hourly_avg.loc[h, 'wind_speed_10m']) if ('wind_speed_10m' in hourly_avg.columns and h in hourly_avg.index) else 0.0
        }
        for h in range(24)
    ]

    # Outlier detection using 99th percentile thresholding
    outliers = {}
    for col in ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']:
        if col in df.columns:
            p99 = float(np.percentile(df[col].dropna(), 99))
            max_val = float(df[col].max())
            outliers[col] = {"max": round(max_val, 2), "p99": round(p99, 2)}

    eda_results = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source,
        "total_observations": len(df),
        "target_correlations": pm25_correlations,
        "full_correlation_matrix": full_corr_matrix_dict,
        "hourly_diurnal_profile": hourly_profile,
        "outlier_thresholds": outliers,
        "statistical_summary": stats_summary
    }

    print("\nStatistical Summary:")
    print(stats_df[['mean', 'std', 'min', '50%', 'max']])

    print("\nPM2.5 Feature Correlations:")
    for feat, corr in pm25_correlations.items():
        print(f"  {feat:22s}: {corr:+.4f}")

    print("\nOutlier Thresholds (99th Percentile):")
    for feat, vals in outliers.items():
        print(f"  {feat:18s} -> Max: {vals['max']:6.2f} | P99: {vals['p99']:6.2f}")

    if save_json:
        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", "eda_summary.json")
        with open(out_path, "w") as f:
            json.dump(eda_results, f, indent=2)
        print(f"\nEDA summary exported to '{out_path}'")

    return eda_results






if __name__ == "__main__":
    run_eda()
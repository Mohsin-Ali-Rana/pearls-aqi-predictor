import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from fetch_raw_data import fetch_historical_aqi
    from config import LOCATION_LATITUDE, LOCATION_LONGITUDE
except ImportError:
    from src.fetch_raw_data import fetch_historical_aqi
    from src.config import LOCATION_LATITUDE, LOCATION_LONGITUDE

def run_eda(save_json: bool = True) -> dict:
    """
    Executes Exploratory Data Analysis (EDA) on raw historical air quality and weather data.
    Computes statistical summaries, target correlations, diurnal patterns, and outlier thresholds.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    start_str = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

    print(f"=== Running Exploratory Data Analysis (EDA) ===")
    print(f"Date Range: {start_str} to {today_str} ({LOCATION_LATITUDE}, {LOCATION_LONGITUDE})")

    df = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, start_str, today_str)
    if df.empty:
        raise ValueError("Failed to fetch historical dataset for EDA.")

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek

    # 1. Statistical Summary
    numeric_cols = [c for c in df.columns if c not in ['time', 'hour', 'day_of_week']]
    stats_df = df[numeric_cols].describe().T.round(2)
    stats_summary = stats_df.to_dict(orient="index")

    # 2. Correlation Matrix with PM2.5 and Full Pairwise Matrix
    corr_matrix = df[numeric_cols].corr().round(4)
    pm25_correlations = corr_matrix['pm2_5'].sort_values(ascending=False).to_dict()
    full_corr_matrix_dict = corr_matrix.to_dict()

    # 3. Diurnal (Hourly) PM2.5 & AQI Profile
    hourly_avg = df.groupby('hour')[['pm2_5', 'european_aqi', 'temperature_2m', 'wind_speed_10m']].mean().round(2)
    hourly_profile = [
        {
            "hour": f"{h:02d}:00",
            "pm25": float(hourly_avg.loc[h, 'pm2_5']) if 'pm2_5' in hourly_avg.columns else 0.0,
            "aqi": float(hourly_avg.loc[h, 'european_aqi']) if 'european_aqi' in hourly_avg.columns else 0.0,
            "temp": float(hourly_avg.loc[h, 'temperature_2m']) if 'temperature_2m' in hourly_avg.columns else 0.0,
            "wind": float(hourly_avg.loc[h, 'wind_speed_10m']) if 'wind_speed_10m' in hourly_avg.columns else 0.0
        }
        for h in range(24)
    ]

    # 4. Outlier Analysis (99th Percentile)
    outliers = {}
    for col in ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']:
        if col in df.columns:
            p99 = float(np.percentile(df[col].dropna(), 99))
            max_val = float(df[col].max())
            outliers[col] = {"max": round(max_val, 2), "p99": round(p99, 2)}

    eda_results = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_observations": len(df),
        "target_correlations": pm25_correlations,
        "full_correlation_matrix": full_corr_matrix_dict,
        "hourly_diurnal_profile": hourly_profile,
        "outlier_thresholds": outliers,
        "statistical_summary": stats_summary
    }

    print("\n--- STATISTICAL SUMMARY ---")
    print(stats_df[['mean', 'std', 'min', '50%', 'max']])

    print("\n--- PM2.5 FEATURE CORRELATIONS ---")
    for feat, corr in pm25_correlations.items():
        print(f"  {feat:22s}: {corr:+.4f}")

    print("\n--- OUTLIER THRESHOLDS (99th Percentile) ---")
    for feat, vals in outliers.items():
        print(f"  {feat:18s} -> Max: {vals['max']:6.2f} | P99: {vals['p99']:6.2f}")

    if save_json:
        os.makedirs("data", exist_ok=True)
        out_path = os.path.join("data", "eda_summary.json")
        with open(out_path, "w") as f:
            json.dump(eda_results, f, indent=2)
        print(f"\n✅ EDA summary exported to '{out_path}'")

    return eda_results

if __name__ == "__main__":
    run_eda()
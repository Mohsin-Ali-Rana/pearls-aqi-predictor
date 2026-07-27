import pandas as pd
import numpy as np
from fetch_raw_data import fetch_historical_aqi
from config import LOCATION_LATITUDE, LOCATION_LONGITUDE

def run_eda():
    # 1. Fetch our raw historical dataset
    df = fetch_historical_aqi(LOCATION_LATITUDE, LOCATION_LONGITUDE, "2026-06-01", "2026-07-21")
    
    print("==================================================")
    print("            STATISTICAL SUMMARY                   ")
    print("==================================================")
    # View mean, std, min, 25%, 50%, 75%, max
    print(df.describe().T)
    
    print("\n==================================================")
    print("            CORRELATION MATRIX                    ")
    print("==================================================")
    # Measure linear relationship between pollutants and AQI
    numeric_df = df.drop(columns=['time'])
    corr_matrix = numeric_df.corr()
    print(corr_matrix['european_aqi'].sort_values(ascending=False))
    
    print("\n==================================================")
    print("            OUTLIER DETECTION (99th Percentile)   ")
    print("==================================================")
    # Identify severe pollution spikes
    for col in ['pm10', 'pm2_5', 'nitrogen_dioxide', 'ozone']:
        p99 = np.percentile(df[col], 99)
        max_val = df[col].max()
        print(f"{col:18s} -> Max: {max_val:6.2f} | 99th Percentile: {p99:6.2f}")

if __name__ == "__main__":
    run_eda()
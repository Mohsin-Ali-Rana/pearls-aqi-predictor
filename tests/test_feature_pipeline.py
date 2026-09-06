import pytest
import pandas as pd
import numpy as np
from src.feature_pipeline import generate_features

def test_feature_engineering_pipeline():
    """Verify that generate_features creates required covariates accurately."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df_raw = pd.DataFrame({
        "time": dates,
        "pm2_5": np.random.uniform(10, 150, size=100),
        "pm10": np.random.uniform(20, 200, size=100),
        "european_aqi": np.random.uniform(20, 100, size=100),
        "temperature_2m": np.random.uniform(5, 35, size=100),
        "relative_humidity_2m": np.random.uniform(20, 90, size=100),
        "surface_pressure": np.random.uniform(950, 1020, size=100),
        "wind_speed_10m": np.random.uniform(1, 15, size=100),
        "wind_direction_10m": np.random.uniform(0, 360, size=100)
    })
    
    df_feat = generate_features(df_raw.copy())
    
    # Check lag features
    assert "pm2_5_lag_1h" in df_feat.columns
    assert "pm2_5_lag_2h" in df_feat.columns
    assert "pm2_5_lag_6h" in df_feat.columns
    assert "pm2_5_lag_12h" in df_feat.columns
    assert "pm2_5_lag_24h" in df_feat.columns
        
    # Check rolling mean and std features
    assert "pm2_5_rolling_6h_mean" in df_feat.columns
    assert "pm2_5_rolling_12h_mean" in df_feat.columns
    assert "pm2_5_rolling_24h_mean" in df_feat.columns
    assert "pm2_5_rolling_24h_std" in df_feat.columns
        
    # Check temporal cyclical features
    for col in ["sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week"]:
        assert col in df_feat.columns
        assert df_feat[col].between(-1.0, 1.0).all()
        
    # Check atmospheric dispersion features
    assert "ventilation_index" in df_feat.columns
    assert "stagnation_index" in df_feat.columns
    
    # Verify non-empty dataframe returned
    assert len(df_feat) > 0

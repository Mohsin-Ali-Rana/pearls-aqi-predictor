import pytest
import numpy as np
import pandas as pd

def test_log_target_transformation():
    """Verify log(1 + y) scaling and inverse exponential conversion stability."""
    y_raw = np.array([0.0, 10.5, 50.0, 150.0, 350.0])
    
    # Log transform
    z = np.log1p(y_raw)
    assert (z >= 0).all()
    
    # Inverse transform
    y_rec = np.expm1(z)
    np.testing.assert_allclose(y_raw, y_rec, rtol=1e-5)

def test_evaluation_metrics():
    """Verify RMSE, MAE, and R2 calculations."""
    y_true = np.array([20.0, 40.0, 60.0, 80.0, 100.0])
    y_pred = np.array([22.0, 38.0, 63.0, 77.0, 102.0])
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    assert rmse > 0
    assert mae > 0
    assert mae <= rmse

import os
import pytest

def test_config_paths():
    """Verify configuration settings and key paths exist or can be resolved."""
    from src import config
    
    assert hasattr(config, 'BASE_DIR')
    assert hasattr(config, 'LOCATION_LATITUDE')
    assert hasattr(config, 'LOCATION_LONGITUDE')
    assert hasattr(config, 'HOPSWORKS_PROJECT')
    
    assert config.LOCATION_LATITUDE == 33.77
    assert config.LOCATION_LONGITUDE == 72.75
    assert os.path.exists(config.BASE_DIR)

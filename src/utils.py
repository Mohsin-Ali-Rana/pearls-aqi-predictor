# PM2.5 concentration (µg/m³) ko US EPA AQI standard scale (0-500) par linear interpolation se map karne ka utility function
def convert_pm25_to_aqi(pm25: float) -> float:
    c = max(0.0, float(pm25))
    if c <= 12.0:
        aqi = (50.0 / 12.0) * c
    elif c <= 35.4:
        aqi = 51.0 + ((49.0 / (35.4 - 12.1)) * (c - 12.1))
    elif c <= 55.4:
        aqi = 101.0 + ((49.0 / (55.4 - 35.5)) * (c - 35.5))
    elif c <= 150.4:
        aqi = 151.0 + ((49.0 / (150.4 - 55.5)) * (c - 55.5))
    elif c <= 250.4:
        aqi = 201.0 + ((99.0 / (250.4 - 150.5)) * (c - 150.5))
    elif c <= 350.4:
        aqi = 301.0 + ((99.0 / (350.4 - 250.5)) * (c - 250.5))
    else:
        aqi = 401.0 + ((99.0 / (500.4 - 350.5)) * (c - 350.5))

    return float(round(min(500.0, max(0.0, aqi)), 1))






# AQI numerical value ke according EPA health category description return karne ka function
def get_aqi_status(aqi_val: float) -> str:
    aqi = float(aqi_val)
    if aqi <= 50.0:
        return "Good"
    elif aqi <= 100.0:
        return "Moderate"
    elif aqi <= 150.0:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200.0:
        return "Unhealthy"
    elif aqi <= 300.0:
        return "Very Unhealthy"
    else:
        return "Hazardous"

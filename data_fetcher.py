import ee
import requests
import random

# Attempt to initialize Google Earth Engine
try:
    ee.Initialize()
    GEE_ACTIVE = True
except Exception as e:
    print("GEE not initialized. Using fallback public APIs.")
    GEE_ACTIVE = False


def fetch_environmental_data(lat: float, lon: float) -> dict:
    if GEE_ACTIVE:
        try:
            return _fetch_from_gee(lat, lon)
        except Exception as e:
            print(f"GEE Fetch failed: {e}. Falling back...")
            return _fetch_from_public_apis(lat, lon)
    else:
        return _fetch_from_public_apis(lat, lon)


def _fetch_from_gee(lat: float, lon: float) -> dict:
    point = ee.Geometry.Point([lon, lat])

    # Elevation (SRTM)
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.sample(point, 30).first().get('elevation').getInfo()

    # Climate (TerraClimate)
    climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('2022-01-01', '2022-12-31').mean()
    temp = climate.select('tmmx').sample(point, 30).first().get('tmmx').getInfo() * 0.1  # scaled
    precip = climate.select('pr').sample(point, 30).first().get('pr').getInfo()

    # Landsat 8 (NDVI & NDWI)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(point).filterDate('2022-06-01',
                                                                                     '2022-09-01').median()
    ndvi = l8.normalizedDifference(['SR_B5', 'SR_B4']).sample(point, 30).first().get('nd').getInfo()
    ndwi = l8.normalizedDifference(['SR_B3', 'SR_B5']).sample(point, 30).first().get('nd').getInfo()

    return {
        "mid_year_temp": temp,
        "precipitation": precip,
        "frost_risk": random.uniform(0, 0.5),  # GEE frost risk requires complex calculation, mocked here
        "elevation": elevation,
        "ndvi": ndvi,
        "ndwi": ndwi
    }


def _fetch_from_public_apis(lat: float, lon: float) -> dict:
    """Fallback if GEE is not authenticated (Open-Meteo API)"""
    meteo_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2023-06-01&end_date=2023-08-31&daily=temperature_2m_mean,precipitation_sum&timezone=auto"
    resp = requests.get(meteo_url).json()

    temp = sum(resp['daily']['temperature_2m_mean']) / len(resp['daily']['temperature_2m_mean'])
    precip = sum(resp['daily']['precipitation_sum'])

    return {
        "mid_year_temp": round(temp, 2),
        "precipitation": round(precip, 2),
        "frost_risk": round(random.uniform(0.0, 0.8), 2), # Mocked for fallback
        "elevation": round(
            requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}").json().get(
                'elevation', [0])[0], 2),
        "ndvi": round(random.uniform(0.3, 0.7), 2),  # Mocked for fallback
        "ndwi": round(random.uniform(-0.2, 0.3), 2)  # Mocked for fallback
    }
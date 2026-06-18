import ee
import requests
import random

try:
    # Replace with your actual project ID
    ee.Initialize(project='pp-2-sem-grapes')
    GEE_ACTIVE = True
except Exception as e:
    print(f"GEE Not Active: {e}")
    GEE_ACTIVE = False


def fetch_environmental_data(lat: float, lon: float) -> dict:
    if GEE_ACTIVE:
        try:
            return _fetch_from_gee(lat, lon)
        except Exception as e:
            print(f"GEE Fetch failed: {e}. Falling back...")
            return _fetch_from_public_apis(lat, lon)
    return _fetch_from_public_apis(lat, lon)


def _fetch_from_gee(lat: float, lon: float) -> dict:
    point = ee.Geometry.Point([lon, lat])

    # 1. Terrain
    dem = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(dem)
    t_info = terrain.sample(point, 30).first().getInfo()

    # Check if point is in ocean/no-data zone
    if not t_info: raise ValueError("No terrain data at these coordinates (Ocean?)")
    t_data = t_info['properties']

    # 2. Climate
    climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate('2023-01-01', '2023-12-31').mean()
    c_info = climate.sample(point, 30).first().getInfo()
    c_data = c_info['properties'] if c_info else {}

    # 3. Satellites
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(point).filterDate('2023-06-01',
                                                                                     '2023-09-01').median()
    ndvi_img = l8.normalizedDifference(['SR_B5', 'SR_B4'])
    ndwi_img = l8.normalizedDifference(['SR_B3', 'SR_B5'])

    ndvi_info = ndvi_img.sample(point, 30).first().getInfo()
    ndwi_info = ndwi_img.sample(point, 30).first().getInfo()

    return {
        "elevation": t_data.get('elevation', 0.0),
        "elevation_status": "GEE_SUCCESS",
        "slope": t_data.get('slope', 0.0),
        "slope_status": "GEE_SUCCESS",
        "aspect": t_data.get('aspect', 0.0),
        "aspect_status": "GEE_SUCCESS",
        "hillshade": t_data.get('hillshade', 0.0),
        "hillshade_status": "GEE_SUCCESS",
        "mid_year_temp": c_data.get('tmmx', 0.0) * 0.1 if 'tmmx' in c_data else 0.0,
        "precipitation": c_data.get('pr', 0.0) if 'pr' in c_data else 0.0,
        "ndvi": ndvi_info['properties']['nd'] if ndvi_info else 0.0,
        "ndwi": ndwi_info['properties']['nd'] if ndwi_info else 0.0
    }


def _fetch_from_public_apis(lat: float, lon: float) -> dict:
    """Fallback ensuring ALL keys exist to prevent Crashes."""
    return {
        "elevation": 0.0,
        "elevation_status": "FAILED",
        "slope": 0.0,
        "slope_status": "FAILED",
        "aspect": 0.0,
        "aspect_status": "FAILED",
        "hillshade": 0.0,
        "hillshade_status": "FAILED",
        "mid_year_temp": 0.0,
        "precipitation": 0.0,
        "ndvi": 0.0,
        "ndwi": 0.0
    }
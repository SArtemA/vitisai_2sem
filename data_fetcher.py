import ee
import requests
import random

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

    # 1. Terrain Data (SRTM)
    dem = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(dem)
    t_data = terrain.sample(point, 30).first().getInfo()['properties']

    # 2. Climate Data (TerraClimate - Average for 2023)
    climate = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE') \
        .filterDate('2023-01-01', '2023-12-31').mean()
    # tmmx is Max Temp, scaled by 0.1
    c_data = climate.sample(point, 30).first().getInfo()['properties']

    # 3. Satellite Imagery (Landsat 8 - Summer 2023 for NDVI)
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(point).filterDate('2023-06-01', '2023-09-01').median()

    # Calculate NDVI: (NIR - Red) / (NIR + Red) -> (B5 - B4)
    ndvi_img = l8.normalizedDifference(['SR_B5', 'SR_B4'])
    # Calculate NDWI: (Green - NIR) / (Green + NIR) -> (B3 - B5)
    ndwi_img = l8.normalizedDifference(['SR_B3', 'SR_B5'])

    ndvi_val = ndvi_img.sample(point, 30).first().getInfo()['properties']['nd']
    ndwi_val = ndwi_img.sample(point, 30).first().getInfo()['properties']['nd']

    return {
        "elevation": t_data.get('elevation', 0.0),
        "elevation_status": "GEE_SUCCESS",
        "slope": t_data.get('slope', 0.0),
        "slope_status": "GEE_SUCCESS",
        "aspect": t_data.get('aspect', 0.0),
        "aspect_status": "GEE_SUCCESS",
        "hillshade": t_data.get('hillshade', 0.0),
        "hillshade_status": "GEE_SUCCESS",
        # New climate/sat data
        "mid_year_temp": c_data.get('tmmx', 0.0) * 0.1,
        "precipitation": c_data.get('pr', 0.0),
        "ndvi": ndvi_val,
        "ndwi": ndwi_val
    }

def _fetch_from_public_apis(lat: float, lon: float) -> dict:
    """Fallback if GEE is not authenticated."""
    # Open-Meteo elevation API fallback
    elev_resp = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}").json()
    elevation = elev_resp.get('elevation', [0])[0]

    return {
        "elevation": round(elevation, 2),
        "elevation_status": "API_FALLBACK",
        "slope": round(random.uniform(0, 15), 2),  # Mocked fallback
        "slope_status": "MOCKED_FALLBACK",
        "aspect": round(random.uniform(0, 360), 2),  # Mocked fallback
        "aspect_status": "MOCKED_FALLBACK",
        "hillshade": round(random.uniform(0, 255), 2),  # Mocked fallback
        "hillshade_status": "MOCKED_FALLBACK"
    }
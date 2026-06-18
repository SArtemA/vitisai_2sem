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
    date_range = {'start': '2023-05-10', 'end': '2023-09-15'}

    # Helper function to query GEE collections safely
    def get_properties(image, scale):
        try:
            sampled = image.sample(point, scale).first()
            if sampled:
                info = sampled.getInfo()
                if info and 'properties' in info:
                    return info['properties']
        except Exception:
            pass
        return {}

    # 1. Terrain (SRTM)
    dem = ee.Image('USGS/SRTMGL1_003')
    terrain_img = ee.Terrain.products(dem)
    terrain = get_properties(terrain_img, 30)

    # 2. Weather & Climate (ERA5-Land Monthly Aggregated)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(date_range['start'], date_range['end']).mean()
    weather_data = get_properties(era5, 30)

    # 3. Soil Data (OpenLandMap)
    soil_ph_img = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b60')
    soil_soc_img = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b60')


    ph_props = get_properties(soil_ph_img, 30)
    soc_props = get_properties(soil_soc_img, 30)

    ph_val = ph_props.get('b60', 60)
    soc_val = soc_props.get('b60', 0)

    # 4. Land Cover (ESA WorldCover 10m)
    land_cover_img = ee.ImageCollection("ESA/WorldCover/v100").first().select('Map')
    lc_props = get_properties(land_cover_img, 10)
    lc_val = lc_props.get('Map', 0)

    # 5. Vegetation Indices (MODIS - MOD13Q1 250m)
    modis_veg = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(date_range['start'], date_range['end']).median()
    veg_props = get_properties(modis_veg, 250)
    ndvi_val = (veg_props.get('NDVI', 0) * 0.0001) if 'NDVI' in veg_props else 0.0
    evi_val = (veg_props.get('EVI', 0) * 0.0001) if 'EVI' in veg_props else 0.0

    # 6. Canopy Water Index (NDWI from MODIS - MOD09A1 500m bands b02 and b06)
    modis_surface = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(date_range['start'], date_range['end']).median()
    ndwi_img = modis_surface.normalizedDifference(['sur_refl_b02', 'sur_refl_b06'])
    ndwi_props = get_properties(ndwi_img, 500)
    ndwi_val = ndwi_props.get('nd', 0.0)

    # 7. LAI (Leaf Area Index)
    lai_img = ee.ImageCollection("MODIS/061/MCD15A3H").filterDate(date_range['start'], date_range['end']).median()
    lai_props = get_properties(lai_img, 500)
    lai_val = (lai_props.get('Lai', 0) * 0.1) if 'Lai' in lai_props else 0.0


    # 8. Fire Risk (Proxy via TerraClimate PDSI)
    fire_proxy = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(date_range['start'],
                                                                            date_range['end']).mean()
    fire_props = get_properties(fire_proxy, 30)
    pdsi = fire_props.get('pdsi', 0.0)

    # 9. Winkler Index (Growing Degree Days, base 10°C)
    if lat >= 0:
        gdd_start, gdd_end = '2023-04-01', '2023-10-31'
    else:
        gdd_start, gdd_end = '2022-10-01', '2023-04-30'

    daily_temp = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
        .filterDate(gdd_start, gdd_end) \
        .select('temperature_2m')

    def calc_gdd(img):
        temp_c = img.subtract(273.15)
        gdd_day = temp_c.subtract(10.0).clamp(0.0, 100.0)
        return gdd_day.copyProperties(img, ['system:time_start'])

    gdd_sum_img = daily_temp.map(calc_gdd).sum()
    gdd_props = get_properties(gdd_sum_img, 30)
    winkler_index = gdd_props.get('temperature_2m', 0.0)

    # Wind Speed Calculation (sqrt(u^2 + v^2))
    u = weather_data.get('u_component_of_wind_10m', 0.0)
    v = weather_data.get('v_component_of_wind_10m', 0.0)
    wind_speed = (u ** 2 + v ** 2) ** 0.5

    mid_year_temp_k = weather_data.get('temperature_2m', 273.15)
    humidity_k = weather_data.get('dewpoint_temperature_2m', 273.15)


    # new parameters
    # --- New OpenLandMap Soil Layers ---
    soil_bulk_dens_img = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b60')
    soil_sand_img = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b60')
    soil_clay_img = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b60')
    soil_class_img = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02").select('b60')

    soil_bulk_dens_props = get_properties(soil_bulk_dens_img, 30)
    soil_sand_props = get_properties(soil_sand_img, 30)
    soil_clay_props = get_properties(soil_clay_img, 30)
    soil_class_props = get_properties(soil_class_img, 30)

    # --- Corrected parameters (removed tuple-making trailing commas) ---
    fpar_val = (lai_props.get('Fpar', 0) * 0.1) if 'Lai' in lai_props else 0.0
    surface_pressure = weather_data.get('surface_pressure', 0.0)  # Pa
    potential_evaporation_sum = weather_data.get('potential_evaporation_sum', 0.0)  # m
    surface_sensible_heat_flux_sum = weather_data.get('surface_sensible_heat_flux_sum', 0.0)  # J/m²
    volumetric_soil_water_layer_3 = weather_data.get('volumetric_soil_water_layer_3', 0.0)  # %
    skin_reservoir_content = weather_data.get('skin_reservoir_content', 0.0)  # m
    soil_temperature_level_3 = weather_data.get('soil_temperature_level_3', 0.0)  # K
    skin_temperature = weather_data.get('skin_temperature', 0.0)  # K

    soil_bulk_dens_val = soil_bulk_dens_props.get('b60', 0.0)
    soil_sand_val = soil_sand_props.get('b60', 0.0)
    soil_clay_val = soil_clay_props.get('b60', 0.0)
    soil_class_val = soil_class_props.get('b60', 0.0)

    return {
        "elevation": terrain.get('elevation', 0.0), #
        "slope": terrain.get('slope', 0.0), #
        "aspect": terrain.get('aspect', 0.0), #
        "hillshade": terrain.get('hillshade', 0.0),#
        "mid_year_temp": mid_year_temp_k - 273.15, #
        "precipitation": weather_data.get('total_precipitation_sum', 0.0) * 1000,
        "humidity": humidity_k - 273.15,
        "solar_radiation": weather_data.get('surface_solar_radiation_downwards_sum', 0.0) / 1000000,
        "wind_speed": wind_speed,
        "evapotranspiration": weather_data.get('total_evaporation_sum', 0.0) * 1000,
        "evi": evi_val,
        "lai": lai_val,
        "land_cover_type": lc_val,
        "soil_ph": ph_val / 10.0,
        "soil_organic_carbon": soc_val,
        "fire_risk": pdsi,
        "winkler_index": winkler_index,
        "ndvi": ndvi_val,
        "ndwi": ndwi_val,
        "elevation_status": "SUCCESS" if 'elevation' in terrain else "FAILED",
        "slope_status": "SUCCESS" if 'slope' in terrain else "FAILED",
        "aspect_status": "SUCCESS" if 'aspect' in terrain else "FAILED",
        "hillshade_status": "SUCCESS" if 'hillshade' in terrain else "FAILED",
        "fpar": fpar_val,
        "surface_pressure": surface_pressure * 1000, # KPa
        "potential_evaporation_sum": potential_evaporation_sum * 1000,
        "surface_sensible_heat_flux_sum": surface_sensible_heat_flux_sum,
        "volumetric_soil_water_layer_3": volumetric_soil_water_layer_3,
        "skin_reservoir_content": skin_reservoir_content * 1000,
        "soil_temperature_level_3": soil_temperature_level_3 - 273.15, # °C
        "skin_temperature": skin_temperature - 273.15, # °C
        "soil_bulk_density": soil_bulk_dens_val,
        "soil_sand": soil_sand_val,
        "soil_clay": soil_clay_val,
        "soil_texture_class": soil_class_val
    }


def _fetch_from_public_apis(lat: float, lon: float) -> dict:
    print("пу пу пу")
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
        "humidity": 0.0,
        "solar_radiation": 0.0,
        "wind_speed": 0.0,
        "evapotranspiration": 0.0,
        "evi": 0.0,
        "lai": 0.0,
        "land_cover_type": 0,
        "soil_ph": 0.0,
        "soil_organic_carbon": 0.0,
        "fire_risk": 0.0,
        "winkler_index": 0.0,
        "ndvi": 0.0,
        "ndwi": 0.0,

        "fpar": 0.0,
        "surface_pressure": 0,
        "potential_evaporation_sum": 0.0,
        "surface_sensible_heat_flux_sum": 0.0,
        "volumetric_soil_water_layer_3": 0.0,
        "skin_reservoir_content": 0.0,
        "soil_temperature_level_3": 0,
        "skin_temperature": 0,
        "soil_bulk_density": 0.0,
        "soil_sand": 0.0,
        "soil_clay": 0.0,
        "soil_texture_class": 0
    }
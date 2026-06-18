from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import os

from databases import database
from models.ml_model import *
import data_fetcher

app = FastAPI(title="Viticulture Predictor")
templates = Jinja2Templates(directory="templates")

variety_model = None
suit_class = None

@app.on_event("startup")
async def startup_event():
    global variety_model, suit_class
    try:
        # ml_model.load_model_if_exists()
        suit_class = BinSuitClassifier()
        if suit_class.model is None:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Предсказания будут недоступны.")
        else:
            print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

    try:
        variety_model = MultiGrapeXGBClassifier()
    except Exception as var_m_e:
        print('Ошибка при загрузке модели:', var_m_e)
        print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Рекомендации будут недоступны.")


class CoordinatesIn(BaseModel):
    lat: float
    lon: float




@app.get("/", response_class=HTMLResponse)
async def map_page(request: Request):
    return templates.TemplateResponse(request=request, name="map.html")


@app.get("/predict_page", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request=request, name="predict.html")


@app.get("/api/vineyards")
def get_vineyards(db: Session = Depends(database.get_db)):
    vineyards = db.query(database.VineyardFeature).all()
    return vineyards



@app.post("/api/predict")
def predict_suitability(coords: CoordinatesIn, db: Session = Depends(database.get_db)):
    if suit_class is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не обучена или не найдена. Запустите pipeline_vitis.py для обучения."
        )

    try:
        env_data = data_fetcher.fetch_environmental_data(coords.lat, coords.lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка получения данных: {e}")
    print(env_data)
    try:
        # print('is_suitable')
        is_suitable = suit_class.predict_suitability(env_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

    recommendations = []
    try:
        print('recommendations')
        if is_suitable:
            # print("Land is suitable. Calculating variety rankings...")
            try:
                if variety_model is not None:
                    recommendations = variety_model.predict(env_data)
                    # print(recommendations)
                else:
                    print('Не удалось получить рекомендации')
            except Exception as recommendations_e:
                print('recommendations_e',recommendations_e)
        else:
            print("Land is unsuitable. Skipping variety analysis.")
    except Exception as g_recommend_e:
        print('g_recommend_e', g_recommend_e)

    new_record = database.VineyardFeature(
        lat=coords.lat,
        lon=coords.lon,
        elevation=env_data["elevation"],
        elevation_GEE_USGS_30m_status=env_data.get("elevation_status"),
        slope=env_data["slope"],
        slope_GEE_USGS_30m_status=env_data.get("slope_status"),
        aspect=env_data["aspect"],
        aspect_GEE_USGS_30m_status=env_data.get("aspect_status"),
        hillshade=env_data["hillshade"],
        hillshade_GEE_USGS_30m_status=env_data.get("hillshade_status"),
        mid_year_temp=env_data["mid_year_temp"],
        precipitation=env_data["precipitation"],
        ndvi=env_data["ndvi"],
        ndwi=env_data["ndwi"],
        solar_radiation=env_data["solar_radiation"],
        humidity=env_data["humidity"],
        wind_speed=env_data["wind_speed"],
        soil_ph=env_data["soil_ph"],
        soil_organic_carbon=env_data["soil_organic_carbon"],
        fire_risk=env_data["fire_risk"],
        evi=env_data["evi"],
        lai=env_data["lai"],
        land_cover_type=env_data["land_cover_type"],
        is_suitable=is_suitable,
        evapotranspiration=env_data["evapotranspiration"],
        fpar=env_data.get("fpar"),
        surface_pressure=env_data.get("surface_pressure"),
        potential_evaporation_sum=env_data.get("potential_evaporation_sum"),
        surface_sensible_heat_flux_sum=env_data.get("surface_sensible_heat_flux_sum"),
        volumetric_soil_water_layer_3=env_data.get("volumetric_soil_water_layer_3"),
        skin_reservoir_content=env_data.get("skin_reservoir_content"),
        soil_temperature_level_3=env_data.get("soil_temperature_level_3"),
        skin_temperature=env_data.get("skin_temperature"),
        soil_bulk_density=env_data.get("soil_bulk_density"),
        soil_sand=env_data.get("soil_sand"),
        soil_clay=env_data.get("soil_clay"),
        soil_texture_class=env_data.get("soil_texture_class")
    )
    db.add(new_record)
    db.commit()

    return {
        "suitable": is_suitable,
        "data_collected": env_data,
        "recommendations": recommendations
    }
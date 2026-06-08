from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import os

from databases import database
from models import ml_model
import data_fetcher

app = FastAPI(title="Viticulture Predictor")
templates = Jinja2Templates(directory="templates")

variety_model = None


@app.on_event("startup")
async def startup_event():
    global variety_model
    try:
        ml_model.load_model_if_exists()
        if ml_model.model is None:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Предсказания будут недоступны.")
        else:
            print("Модель успешно загружена.")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")

    try:
        variety_model = ml_model.GrapeXGBClassifier()
    except Exception as var_m_e:
        print('Ошибка при загрузке модели:', var_m_e)
        print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Рекомендации будут недоступны.")


class CoordinatesIn(BaseModel):
    lat: float
    lon: float


class VineyardOut(BaseModel):
    id: int
    lat: float
    lon: float
    elevation_GEE_USGS_30m: float | None
    slope_GEE_USGS_30m: float | None
    aspect_GEE_USGS_30m: float | None
    hillshade_GEE_USGS_30m: float | None
    mid_year_temp: float | None
    precipitation: float | None
    ndvi: float | None
    ndwi: float | None
    is_suitable: bool | None

    class Config:
        from_attributes = True


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
    if ml_model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не обучена или не найдена. Запустите pipeline_vitis.py для обучения."
        )

    try:
        env_data = data_fetcher.fetch_environmental_data(coords.lat, coords.lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка получения данных: {e}")

    try:
        is_suitable = ml_model.predict_suitability(env_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

    recommendations = []
    try:
        if is_suitable:
            print("Land is suitable. Calculating variety rankings...")
            if variety_model is not None:
                recommendations = variety_model.predict(env_data)
                print(recommendations)
            else:
                print('Не удалось получить рекомендации')
        else:
            print("Land is unsuitable. Skipping variety analysis.")
    except Exception as recommend_e:
        print('recommend_e', recommend_e)

    new_record = database.VineyardFeature(
        lat=coords.lat,
        lon=coords.lon,
        elevation_GEE_USGS_30m=env_data["elevation"],
        elevation_GEE_USGS_30m_status=env_data.get("elevation_status"),
        slope_GEE_USGS_30m=env_data["slope"],
        slope_GEE_USGS_30m_status=env_data.get("slope_status"),
        aspect_GEE_USGS_30m=env_data["aspect"],
        aspect_GEE_USGS_30m_status=env_data.get("aspect_status"),
        hillshade_GEE_USGS_30m=env_data["hillshade"],
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
        is_suitable=is_suitable
    )
    db.add(new_record)
    db.commit()

    return {
        "suitable": is_suitable,
        "data_collected": env_data,
        "recommendations": recommendations
    }
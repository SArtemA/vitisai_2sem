from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
import os

import database
import ml_model          # переписанный модуль, см. ниже
import data_fetcher

app = FastAPI(title="Viticulture Predictor")
templates = Jinja2Templates(directory="templates")

variety_model = None
# ---------- Событие запуска ----------
@app.on_event("startup")
async def startup_event():
    global variety_model
    """При старте пытаемся загрузить модель, но не обучаем её."""
    try:
        ml_model.load_model_if_exists()
        if ml_model.model is None:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Предсказания будут недоступны.")
        else:
            print("Модель успешно загружена.")

    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        # Приложение продолжит работать, но /api/predict будет возвращать ошибку
    try:
        variety_model = ml_model.GrapeXGBClassifier()
    except Exception as var_m_e:
        print('Ошибка при загрузке модели:', 'var_m_e', var_m_e)
        print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Рекомендации будут недоступны.")

# ---------- Модели данных ----------
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
        from_attributes = True  # чтобы работало с объектами SQLAlchemy

# ---------- FRONTEND ROUTES ----------
@app.get("/", response_class=HTMLResponse)
async def map_page(request: Request):
    return templates.TemplateResponse(request=request, name="map.html")

@app.get("/predict_page", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request=request, name="predict.html")

# ---------- BACKEND API ROUTES ----------
@app.get("/api/vineyards")#, response_model=List[VineyardOut])
def get_vineyards(db: Session = Depends(database.get_db)):
    vineyards = db.query(database.VineyardFeature).all()
    return vineyards   # FastAPI автоматически сериализует благодаря response_model


@app.post("/api/predict")
def predict_suitability(coords: CoordinatesIn, db: Session = Depends(database.get_db)):
    # 1. Проверяем, загружена ли модель
    if ml_model.model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель не обучена или не найдена. Запустите pipeline_vitis.py для обучения."
        )

    # 2. Получаем экологические данные
    try:
        env_data = data_fetcher.fetch_environmental_data(coords.lat, coords.lon)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка получения данных: {e}")

    # 3. Предсказание
    try:
        is_suitable = ml_model.predict_suitability(env_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

    recommendations = []
    try:


        # 3. Gate 2: ONLY if suitable, run the variety ranking model
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


    # 4. Сохраняем запрос в БД
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
        is_suitable=is_suitable
    )
    db.add(new_record)
    db.commit()

    return {
        "suitable": is_suitable,
        "data_collected": env_data,
        "recommendations": recommendations  # Will be empty if unsuitable
    }
from pathlib import Path
import threading

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy.orm import Session

import data_fetcher
import retrain
import notifications
from databases import database
from models.ml_model import BinSuitClassifier, MultiGrapeXGBClassifier

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
        print("Ошибка при загрузке модели:", var_m_e)
        print("ПРЕДУПРЕЖДЕНИЕ: Модель не найдена. Рекомендации будут недоступны.")


class CoordinatesIn(BaseModel):
    lat: float
    lon: float


NEW_RECORDS_THRESHOLD = 1000
new_records_counter = 0
_retraining_lock = threading.Lock()
COUNTER_PATH = Path(__file__).resolve().parent / "databases" / "counter.txt"


def counter_load():
    """Increase and persist the number of records collected since last retraining."""
    global new_records_counter
    try:
        current = int(COUNTER_PATH.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        current = 0
    new_records_counter = current + 1
    temporary = COUNTER_PATH.with_suffix(".txt.tmp")
    temporary.write_text(str(new_records_counter), encoding="utf-8")
    temporary.replace(COUNTER_PATH)
    return new_records_counter


def check_and_retrain():
    """Run one retraining job at a time and report its outcome to the developer."""
    global new_records_counter
    with _retraining_lock:
        retraining_started = False
        try:
            new_records_counter = counter_load()
            if new_records_counter < NEW_RECORDS_THRESHOLD:
                return

            retraining_started = True
            print("Достигнут порог новых данных. Запуск фонового переобучения...")
            result = retrain.retrain_pipeline()

            if result.get("status") == "success":
                try:
                    if variety_model is None:
                        raise RuntimeError("Экземпляр MultiGrapeXGBClassifier не инициализирован.")
                    variety_model.load_model()
                except Exception as reload_error:
                    # The artifacts were saved, but the running process could not reload them.
                    result = {
                        "status": "error",
                        "message": f"Модель сохранена, но не удалось загрузить её в приложение: {reload_error}",
                    }

            notifications.process_retraining_result(result)
            print(f"Результат переобучения: {result}")
        except Exception as exc:
            error_result = {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
            notifications.process_retraining_result(error_result)
            print(f"Ошибка фонового переобучения: {exc}")
        finally:
            if retraining_started:
                # The current dataset was already considered by this training attempt.
                # New requests will start accumulating from zero for the next cycle.
                new_records_counter = 0
                temporary = COUNTER_PATH.with_suffix(".txt.tmp")
                temporary.write_text("0", encoding="utf-8")
                temporary.replace(COUNTER_PATH)


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
def predict_suitability(
    coords: CoordinatesIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(database.get_db),
):
    if suit_class is None:
        raise HTTPException(
            status_code=503, detail="Модель не обучена или не найдена. Запустите pipeline_vitis.py для обучения."
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
        print("recommendations")
        if is_suitable:
            # print("Land is suitable. Calculating variety rankings...")
            try:
                if variety_model is not None:
                    recommendations = variety_model.predict(env_data)
                    # print(recommendations)
                else:
                    print("Не удалось получить рекомендации")
            except Exception as recommendations_e:
                print("recommendations_e", recommendations_e)
        else:
            print("Land is unsuitable. Skipping variety analysis.")
    except Exception as g_recommend_e:
        print("g_recommend_e", g_recommend_e)

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
        soil_texture_class=env_data.get("soil_texture_class"),
    )
    db.add(new_record)
    db.commit()

    background_tasks.add_task(check_and_retrain)

    return {"suitable": is_suitable, "data_collected": env_data, "recommendations": recommendations}

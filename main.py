from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database, ml_model, data_fetcher
from fastapi import FastAPI, Request, Depends
import os

app = FastAPI(title="Viticulture Predictor")
templates = Jinja2Templates(directory="templates")





@app.on_event("startup")
async def startup_event():
    # If the model doesn't exist, build it before the first user arrives
    if not os.path.exists(ml_model.MODEL_PATH):
        print("Startup: No model found. Initializing training...")
        # ml_model.train_model()

class CoordinatesIn(BaseModel):
    lat: float
    lon: float


# --- FRONTEND ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def map_page(request: Request):
    return templates.TemplateResponse(request=request, name="map.html")

@app.get("/predict_page", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse(request=request, name="predict.html")


# --- BACKEND API ROUTES ---
@app.get("/api/vineyards")
def get_vineyards(db: Session = Depends(database.get_db)):
    # Query all records from the new V2 table
    vineyards = db.query(database.VineyardFeature).all()
    return vineyards


@app.post("/api/predict")
def predict_suitability(coords: CoordinatesIn, db: Session = Depends(database.get_db)):
    env_data = data_fetcher.fetch_environmental_data(coords.lat, coords.lon)
    is_suitable = ml_model.predict_suitability(env_data)

    new_record = database.VineyardFeature(
        lat=coords.lat,
        lon=coords.lon,
        elevation_GEE_USGS_30m=env_data["elevation"],
        elevation_GEE_USGS_30m_status=env_data["elevation_status"],
        slope_GEE_USGS_30m=env_data["slope"],
        slope_GEE_USGS_30m_status=env_data["slope_status"],
        aspect_GEE_USGS_30m=env_data["aspect"],
        aspect_GEE_USGS_30m_status=env_data["aspect_status"],
        hillshade_GEE_USGS_30m=env_data["hillshade"],
        hillshade_GEE_USGS_30m_status=env_data["hillshade_status"],
        # SAVE NEW COLUMNS
        mid_year_temp=env_data["mid_year_temp"],
        precipitation=env_data["precipitation"],
        ndvi=env_data["ndvi"],
        ndwi=env_data["ndwi"],
        is_suitable=is_suitable
    )
    db.add(new_record)
    db.commit()
    return {"suitable": is_suitable, "data_collected": env_data}
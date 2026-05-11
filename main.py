from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import database, ml_model, data_fetcher

app = FastAPI(title="Viticulture Predictor")
templates = Jinja2Templates(directory="templates")


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
    vineyards = db.query(database.Vineyard).all()
    return vineyards


@app.post("/api/predict")
def predict_suitability(coords: CoordinatesIn, db: Session = Depends(database.get_db)):
    # 1. Fetch environmental data
    env_data = data_fetcher.fetch_environmental_data(coords.lat, coords.lon)

    # 2. Predict using XGBoost
    is_suitable = ml_model.predict_suitability(env_data)

    # 3. Save everything to Database for future model improvement
    new_record = database.Vineyard(
        lat=coords.lat,
        lon=coords.lon,
        mid_year_temp=env_data["mid_year_temp"],
        precipitation=env_data["precipitation"],
        frost_risk=env_data["frost_risk"],
        elevation=env_data["elevation"],
        ndvi=env_data["ndvi"],
        ndwi=env_data["ndwi"],
        is_suitable=is_suitable
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)

    return {"suitable": is_suitable, "data_collected": env_data}
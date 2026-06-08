import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Resolve path relative to this file to keep the db in the databases folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'vineyards_v3.db')}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class VineyardFeature(Base):
    __tablename__ = "vineyard_features"

    osm_id = Column(Integer, primary_key=True, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Terrain
    elevation_GEE_USGS_30m = Column(Float)
    elevation_GEE_USGS_30m_status = Column(String)
    slope_GEE_USGS_30m = Column(Float)
    slope_GEE_USGS_30m_status = Column(String)
    aspect_GEE_USGS_30m = Column(Float)
    aspect_GEE_USGS_30m_status = Column(String)
    hillshade_GEE_USGS_30m = Column(Float)
    hillshade_GEE_USGS_30m_status = Column(String)

    # Climate/Vegetation
    mid_year_temp = Column(Float)
    precipitation = Column(Float)
    ndvi = Column(Float)
    ndwi = Column(Float)
    # --- New GEE Parameters ---
    solar_radiation = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(Float)
    evapotranspiration = Column(Float)
    evi = Column(Float)
    lai = Column(Float)
    fractional_cover = Column(Float)
    land_cover_type = Column(Integer)
    soil_ph = Column(Float)
    soil_organic_carbon = Column(Float)
    fire_risk = Column(Float)

    is_suitable = Column(Boolean, default=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# You might want to delete your old vineyards.db file before running this,
# or change the name below so it creates a fresh database.
SQLALCHEMY_DATABASE_URL = "sqlite:///./vineyard_features.db"
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

    elevation_GEE_USGS_30m = Column(Float)
    elevation_GEE_USGS_30m_status = Column(String)
    slope_GEE_USGS_30m = Column(Float)
    slope_GEE_USGS_30m_status = Column(String)
    aspect_GEE_USGS_30m = Column(Float)
    aspect_GEE_USGS_30m_status = Column(String)
    hillshade_GEE_USGS_30m = Column(Float)
    hillshade_GEE_USGS_30m_status = Column(String)

    # NEW COLUMNS
    mid_year_temp = Column(Float)
    precipitation = Column(Float)
    ndvi = Column(Float)
    ndwi = Column(Float)

    is_suitable = Column(Boolean, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
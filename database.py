from sqlalchemy import create_engine, Column, Integer, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./vineyards.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Vineyard(Base):
    __tablename__ = "vineyards"

    id = Column(Integer, primary_key=True, index=True)
    lat = Column(Float, index=True)
    lon = Column(Float, index=True)
    mid_year_temp = Column(Float)
    precipitation = Column(Float)
    frost_risk = Column(Float)
    elevation = Column(Float)
    ndvi = Column(Float)
    ndwi = Column(Float)
    is_suitable = Column(Boolean)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
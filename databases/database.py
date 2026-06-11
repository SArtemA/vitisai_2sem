import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, text, inspect
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# Place the database file relative to this module
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
    elevation = Column(Float)
    elevation_GEE_USGS_30m_status = Column(String)
    slope = Column(Float)
    slope_GEE_USGS_30m_status = Column(String)
    aspect = Column(Float)
    aspect_GEE_USGS_30m_status = Column(String)
    hillshade = Column(Float)
    hillshade_GEE_USGS_30m_status = Column(String)
    # Climate/Vegetation
    mid_year_temp = Column(Float)
    precipitation = Column(Float)
    ndvi = Column(Float)
    ndwi = Column(Float)
    # GEE Parameters
    solar_radiation = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    # wind_direction = Column(Float)
    evapotranspiration = Column(Float)
    evi = Column(Float)
    lai = Column(Float)
    # fractional_cover = Column(Float)
    land_cover_type = Column(Integer)
    soil_ph = Column(Float)
    soil_organic_carbon = Column(Float)
    fire_risk = Column(Float)

    is_suitable = Column(Boolean, default=True)
    # --- New GEE Parameters (Expanded) ---
    fpar = Column(Float)
    surface_pressure = Column(Float)
    potential_evaporation_sum = Column(Float)
    surface_sensible_heat_flux_sum = Column(Float)
    volumetric_soil_water_layer_3 = Column(Float)
    skin_reservoir_content = Column(Float)
    soil_temperature_level_3 = Column(Float)
    skin_temperature = Column(Float)
    soil_bulk_density = Column(Float)
    soil_sand = Column(Float)
    soil_clay = Column(Float)
    soil_texture_class = Column(Integer)


# 1. Create table structure if it doesn't exist
Base.metadata.create_all(bind=engine)


# 2. Automatically apply missing column migrations to on-disk database
def auto_migrate_schema():
    inspector = inspect(engine)
    table_name = "vineyard_features"

    if table_name in inspector.get_table_names():
        # Get columns that exist on disk
        columns_on_disk = {col["name"] for col in inspector.get_columns(table_name)}
        # Get columns declared in the SQLAlchemy class
        columns_in_model = {col.key for col in VineyardFeature.__table__.columns}

        missing_columns = columns_in_model - columns_on_disk

        if missing_columns:
            print(f"Schema mismatch detected in '{table_name}'. Synchronizing columns...")
            with engine.begin() as conn:  # engine.begin() handles transaction commits automatically
                for col_name in missing_columns:
                    col_obj = VineyardFeature.__table__.columns[col_name]

                    # Deduce column SQL type
                    type_str = str(col_obj.type).upper()
                    sql_type = "FLOAT"
                    if "INT" in type_str:
                        sql_type = "INTEGER"
                    elif "BOOL" in type_str:
                        sql_type = "BOOLEAN"
                    elif "VARCHAR" in type_str or "STR" in type_str:
                        sql_type = "VARCHAR"

                    # Append missing column
                    alter_statement = f'ALTER TABLE {table_name} ADD COLUMN "{col_name}" {sql_type};'
                    conn.execute(text(alter_statement))
                    print(f"  Added column: '{col_name}' ({sql_type})")


try:
    auto_migrate_schema()
except Exception as e:
    print(f"Auto-migration warning: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
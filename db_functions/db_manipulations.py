import sqlite3
import os
import data_fetcher  # Your GEE module
import database  # Your SQLAlchemy models
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Paths
OLD_DB_FILE = "vineyard_features.db"
NEW_DB_FILE = "vineyards_v2.db"


def run_backfill():
    # --- STEP 1: FORCE CREATE THE NEW DATABASE ---
    print(f"Initializing new database: {NEW_DB_FILE}")
    # This line ensures the .db file is created and tables are mapped
    database.Base.metadata.create_all(bind=database.engine)

    if not os.path.exists(OLD_DB_FILE):
        print(f"Old database {OLD_DB_FILE} not found. Nothing to migrate.")
        return

    # --- STEP 2: CONNECT TO OLD AND NEW ---
    old_conn = sqlite3.connect(OLD_DB_FILE)
    old_cursor = old_conn.cursor()

    # We use SQLAlchemy for the new DB to ensure data types are handled correctly
    new_db_session: Session = database.SessionLocal()

    # --- STEP 3: GET OLD DATA ---
    # We select lat/lon. If the old DB had some terrain data, we take that too.
    old_cursor.execute("SELECT lat, lon FROM vineyard_features LIMIT 1000;")
    rows = old_cursor.fetchall()

    print(f"Found {len(rows)} records in old database. Starting enrichment...")

    for i, (lat, lon) in enumerate(rows):
        # Check if this coordinate already exists in the NEW database to avoid duplicates
        exists = new_db_session.query(database.VineyardFeature).filter_by(lat=lat, lon=lon).first()

        # If it exists and already has new features (like ndvi), we skip it
        if exists and exists.ndvi is not None:
            print(f"[{i + 1}/{len(rows)}] Skipping {lat}, {lon} (Already enriched).")
            continue

        print(f"[{i + 1}/{len(rows)}] Enriching data for {lat}, {lon}...")

        try:
            # Fetch ALL 8 features from GEE (Terrain + Climate + Satellite)
            env = data_fetcher.fetch_environmental_data(lat, lon)

            if exists:
                # Update existing record in V2 if it was incomplete
                exists.elevation_GEE_USGS_30m = env["elevation"]
                exists.slope_GEE_USGS_30m = env["slope"]
                exists.aspect_GEE_USGS_30m = env["aspect"]
                exists.hillshade_GEE_USGS_30m = env["hillshade"]
                exists.mid_year_temp = env["mid_year_temp"]
                exists.precipitation = env["precipitation"]
                exists.ndvi = env["ndvi"]
                exists.ndwi = env["ndwi"]
                exists.is_suitable = True  # Assume existing ones are good
            else:
                # Create brand new record in V2
                new_v = database.VineyardFeature(
                    lat=lat, lon=lon,
                    elevation_GEE_USGS_30m=env["elevation"],
                    elevation_GEE_USGS_30m_status=env["elevation_status"],
                    slope_GEE_USGS_30m=env["slope"],
                    slope_GEE_USGS_30m_status=env["slope_status"],
                    aspect_GEE_USGS_30m=env["aspect"],
                    aspect_GEE_USGS_30m_status=env["aspect_status"],
                    hillshade_GEE_USGS_30m=env["hillshade"],
                    hillshade_GEE_USGS_30m_status=env["hillshade_status"],
                    mid_year_temp=env["mid_year_temp"],
                    precipitation=env["precipitation"],
                    ndvi=env["ndvi"],
                    ndwi=env["ndwi"],
                    is_suitable=True
                )
                new_db_session.add(new_v)

            new_db_session.commit()
        except Exception as e:
            print(f"Error enriching {lat}, {lon}: {e}")
            new_db_session.rollback()

    new_db_session.close()
    old_conn.close()
    print(f"✅ Migration and Enrichment Complete! New DB: {NEW_DB_FILE}")


if __name__ == "__main__":
    run_backfill()
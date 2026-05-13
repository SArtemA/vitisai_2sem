import random
import time
import database
import data_fetcher
from sqlalchemy.orm import Session

# Settings
TOTAL_SAMPLES = 1000
DB_FILE = "vineyards_v2.db"


def get_random_unsuitable_coord():
    """
    Generates coordinates in regions generally unsuitable for viticulture:
    - Very high latitudes (Arctic/Antarctic)
    - Equatorial regions (Too tropical/humid)
    - Extreme Longitudes
    """
    # 30% chance: High North (Siberia, Northern Canada)
    # 30% chance: Tropics (Amazon, Sahara, Indonesia)
    # 40% chance: High South or random mountains
    pick = random.random()

    if pick < 0.3:
        lat = random.uniform(60, 75)  # Arctic Tundra
    elif pick < 0.6:
        lat = random.uniform(-10, 10)  # Tropical/Equatorial
    else:
        lat = random.uniform(-60, -40)  # Sub-antarctic

    lon = random.uniform(-180, 180)
    return lat, lon


def run_negative_generation():
    # Ensure database tables exist
    database.Base.metadata.create_all(bind=database.engine)
    db: Session = database.SessionLocal()

    print(f"Starting generation of {TOTAL_SAMPLES} negative samples...")

    success_count = 0
    attempt_count = 0

    while success_count < TOTAL_SAMPLES:
        lat, lon = get_random_unsuitable_coord()

        try:
            env = data_fetcher.fetch_environmental_data(lat, lon)

            # --- IMPROVED SKIP LOGIC ---
            # If GEE failed or hit an ocean, we skip it.
            # We want land data that is BAD for grapes, not just empty water.
            if env.get("elevation") == 0 and env.get("ndvi") == 0:
                continue

            new_record = database.VineyardFeature(
                lat=lat,
                lon=lon,
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
                is_suitable=False
            )

            db.add(new_record)
            db.commit()
            success_count += 1
            if success_count % 10 == 0:
                print(f"✅ Generated {success_count}/{TOTAL_SAMPLES} samples...")

        except Exception as e:
            print(e)
            db.rollback()
            continue

    db.close()
    print(f"🏁 Finished! Added {success_count} negative samples to {DB_FILE}.")


if __name__ == "__main__":
    run_negative_generation()
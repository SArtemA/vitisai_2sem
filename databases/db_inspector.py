import sys
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Ensure the project root directory is on the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from databases.database import Base, engine, VineyardFeature


def print_db_structure():
    """Print the database structure (tables and columns)"""
    inspector = inspect(engine)

    print("=== DATABASE STRUCTURE ===\n")

    tables = inspector.get_table_names()
    print(f"Tables in database: {tables}\n")

    for table_name in tables:
        print(f"Table: {table_name}")
        print("-" * 30)

        columns = inspector.get_columns(table_name)
        for column in columns:
            col_name = column['name']
            col_type = column['type']
            col_nullable = "NULL" if column['nullable'] else "NOT NULL"
            col_default = f"DEFAULT {column['default']}" if column['default'] else ""

            print(f"  {col_name}: {col_type} {col_nullable} {col_default}")
        print()


def print_sample_data(n=5):
    """Print first n records from the vineyard_features table"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        print(f"=== SAMPLE DATA (first {n} records) ===\n")

        records = db.query(VineyardFeature).limit(n).all()

        if not records:
            print("No data found in the vineyard_features table.")
            return

        for i, record in enumerate(records, 1):
            print(f"Record {i}:")
            print("-" * 20)

            print(f"  OSM ID: {record.osm_id}")
            print(f"  Location: {record.lat}, {record.lon}")
            print(f"  Created: {record.created_at}")
            print(f"  Suitable: {record.is_suitable}")

            print(f"  Elevation: {record.elevation_GEE_USGS_30m} ({record.elevation_GEE_USGS_30m_status})")
            print(f"  Slope: {record.slope_GEE_USGS_30m} ({record.slope_GEE_USGS_30m_status})")
            print(f"  Aspect: {record.aspect_GEE_USGS_30m} ({record.aspect_GEE_USGS_30m_status})")

            print(f"  Mid-year temp: {record.mid_year_temp}")
            print(f"  Precipitation: {record.precipitation}")
            print(f"  NDVI: {record.ndvi}")
            print(f"  NDWI: {record.ndwi}")
            print()

    except Exception as e:
        print(f"Error reading data: {e}")
    finally:
        db.close()


def print_db_stats():
    """Print basic statistics about the database"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        print("=== DATABASE STATISTICS ===\n")

        total_count = db.query(VineyardFeature).count()
        print(f"Total records: {total_count}")

        suitable_count = db.query(VineyardFeature).filter(VineyardFeature.is_suitable == True).count()
        unsuitable_count = total_count - suitable_count
        print(f"Suitable: {suitable_count}")
        print(f"Unsuitable: {unsuitable_count}")

        first_record = db.query(VineyardFeature).order_by(VineyardFeature.created_at).first()
        last_record = db.query(VineyardFeature).order_by(VineyardFeature.created_at.desc()).first()

        if first_record and last_record:
            print(f"Date range: {first_record.created_at} to {last_record.created_at}")

    except Exception as e:
        print(f"Error calculating statistics: {e}")
    finally:
        db.close()


def count_suitable_unsuitable():
    """Returns counts of suitable and unsuitable locations"""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        suitable_count = db.query(VineyardFeature).filter(VineyardFeature.is_suitable == True).count()
        unsuitable_count = db.query(VineyardFeature).filter(VineyardFeature.is_suitable == False).count()

        print(f"Подходящие участки (is_suitable=True): {suitable_count}")
        print(f"Неподходящие участки (is_suitable=False): {unsuitable_count}")

        return {
            "suitable": suitable_count,
            "unsuitable": unsuitable_count
        }
    except Exception as e:
        print(f"Ошибка при подсчёте: {e}")
        return None
    finally:
        db.close()


def main():
    print("Starting database inspection...\n")
    count_suitable_unsuitable()
    print_db_stats()
    print("Database inspection completed.")


if __name__ == "__main__":
    main()
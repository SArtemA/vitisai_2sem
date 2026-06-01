# pipeline для заполнения датасета
# Папка с функциями
from services.terrain_GEE_USGS import terrain_GEE_USGS
from services.climate_GEE_TERRA import climate_GEE_TERRA
from services.landsat_GEE import landsat_GEE

# Папка по работе с БД
import db.repository as repository

# Библиотеки
import time
from pathlib import Path
import ee
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout  # Теперь лог будет идти как обычный текст (белым)
)


# Настройки
BATCH_SIZE = 1000
SLEEP_TIME = 0.5

# Путь к БД
db_folder_path = Path(__file__).resolve().parent / "data"
db_folder_path.mkdir(exist_ok=True)
db_name = 'vineyard_1.db'
db_path = db_folder_path / db_name

used_tables = ['vineyard_features', 'negative_features']
used_tables = ['negative_features']


def run_pipeline():
    # Запуск GEE
    ee.Authenticate()
    ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))

    total_processed = 0

    # terrain_GEE_USGS
    # 'elevation_GEE_USGS_30m'
    for table_name in used_tables:
        # Для визуала
        general_row_count = repository.get_count_row(db_path, table_name)
        print(f"*** В таблице '{table_name}' всего позиций - {general_row_count} ***")

        # terrain_GEE_USGS
        print("=====terrain_GEE_USGS=====")
        already_processed = get_already_processed(db_path, table_name, 'elevation_GEE_USGS_30m')
        print(f"Уже обработано строк для terrain_GEE_USGS - {already_processed}", end='')
        print(f" (соотношение {already_processed/general_row_count*100:.2f}%)")

        while False:
            cycle_processed = 0
            # Запрос из БД
            pending_items = repository.get_row_by_status(
                db_path,
                table_name,
                "elevation_GEE_USGS_30m",
                limit=BATCH_SIZE
                )

            # Проверка что данные еще есть
            if not pending_items:
                logging.info("Для terrain_GEE_USGS больше нету не заполненных данных. Выход.")
                break
            else:
                cycle_processed += len(pending_items)
                total_processed += cycle_processed

            # Подгонка данных для функций
            osm_id = [item[0] for item in pending_items]
            coords = [list(item[1:]) for item in pending_items]
            features = create_feature_GEE(coords)

            #
            data_to_db = terrain_GEE_USGS(features, verbose=True)

            #
            already_processed += len(osm_id)
            if repository.update_vineyard_features(
                db_path,
                table_name,
                osm_id,
                data_to_db
                ):
                logging.info(f"Таблица '{table_name}' | 'terrain_GEE_USGS' | {general_row_count}\t-\t{already_processed}\t({already_processed / general_row_count * 100:.2f}%)")

            time.sleep(SLEEP_TIME)
            # break

        # climate_GEE_TERRA
        print("=====climate_GEE_TERRA=====")
        already_processed = get_already_processed(db_path, table_name, 'precip_total_GEE_TERRA_2024')
        print(f"Уже обработано строк для climate_GEE_TERRA - {already_processed}", end='')
        print(f" (соотношение {already_processed / general_row_count * 100:.2f}%)")

        while False:
            cycle_processed = 0
            # Запрос из БД
            pending_items = repository.get_row_by_status(
                db_path,
                table_name,
                ["precip_total_GEE_TERRA_2024", "elevation_GEE_USGS_30m"],
                status=['pending', 'done'],
                limit=BATCH_SIZE,
                random_order=True
                )

            # Проверка что данные еще есть
            if not pending_items:
                logging.info("Для climate_GEE_TERRA больше нету не заполненных данных. Выход.")
                break
            else:
                cycle_processed += len(pending_items)
                total_processed += cycle_processed

            # Подгонка данных для функций
            osm_id = [item[0] for item in pending_items]
            coords = [list(item[1:]) for item in pending_items]
            features = create_feature_GEE(coords)

            #
            data_to_db = climate_GEE_TERRA(
                features,
                year=2024,
                verbose=False
                )

            #
            already_processed += len(osm_id)
            if repository.update_vineyard_features(
                db_path,
                table_name,
                osm_id,
                data_to_db
                ):
                logging.info(f"Таблица '{table_name}' | 'climate_GEE_TERRA' | {general_row_count}\tиз\t{already_processed}\t({already_processed / general_row_count * 100:.2f}%)")

            time.sleep(SLEEP_TIME)
            # break

        # landsat_GEE
        print("=====landsat_GEE=====")
        already_processed = get_already_processed(db_path, table_name, 'NDVI_phase2_2024')
        print(f"Уже обработано строк для landsat_GEE - {already_processed}", end='')
        print(f" (соотношение {already_processed / general_row_count * 100:.2f}%)")

        while True:
            cycle_processed = 0
            # Запрос из БД
            pending_items = repository.get_row_by_status(
                db_path,
                table_name,
                ['NDVI_phase2_2024', 'precip_total_GEE_TERRA_2024', 'elevation_GEE_USGS_30m'],
                status=['pending', 'done', 'done'],
                limit=100
            )

            # Проверка что данные еще есть
            if not pending_items:
                logging.info("Для landsat_GEE больше нету не заполненных данных. Выход.")
                break
            else:
                cycle_processed += len(pending_items)
                total_processed += cycle_processed

            # Подгонка данных для функций
            osm_id = [item[0] for item in pending_items]
            coords = [list(item[1:]) for item in pending_items]
            features = create_feature_GEE(coords)

            #
            data_to_db = landsat_GEE(
                input_data=features,
                year=2024,
                scale=30,
                verbose=True,
                full_output=False
            )

            #
            already_processed += len(osm_id)
            if repository.update_vineyard_features(
                    db_path,
                    table_name,
                    osm_id,
                    data_to_db
            ):
                logging.info(
                    f"Таблица '{table_name}' | 'landsat_GEE' | {general_row_count}\tиз\t{already_processed}\t({already_processed / general_row_count * 100:.2f}%)")

            time.sleep(SLEEP_TIME)
            # break
        # break

def create_feature_GEE(coords: list):
    return ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords])


def get_already_processed(db_path, table_name, col):
    return (
            repository.get_count_row_by_status(
                db_path,
                table_name,
                col,
                status='done'
            ) +
            repository.get_count_row_by_status(
                db_path,
                table_name,
                col,
                status='error'
            )
    )


def set_cols(db_path):
    added_cols = [
        'elevation_GEE_USGS_30m',
        'slope_GEE_USGS_30m',
        'aspect_GEE_USGS_30m',
        'hillshade_GEE_USGS_30m',
        #
        'precip_total_GEE_TERRA_2024',
        'solar_rad_mean_GEE_TERRA_2024',
        'tmax_mean_GEE_TERRA_2024',
        'tmin_mean_GEE_TERRA_2024',
        'water_deficit_total_GEE_TERRA_2024',
        'winkler_gdd_total_GEE_TERRA_2024',
        #
        'NDVI_phase1_2024',
        'NDVI_phase2_2024',
        'NDVI_phase3_2024',
        'NDWI_phase2_2024',
        'SAVI_phase2_2024',
        'SR_B2_mean_2024',
        'SR_B3_mean_2024',
        'SR_B4_mean_2024',
        'SR_B5_mean_2024',
        'SR_B6_mean_2024',
        'SR_B7_mean_2024',
        'ST_B10_mean_2024',
        'cloud_cover_phase2_2024'
    ]

    for table_name in used_tables:
        repository.create_feature_cols(db_path, table_name, added_cols)

if __name__ == "__main__":
    # Колонки
    set_cols(db_path)

    # 1
    run_pipeline()

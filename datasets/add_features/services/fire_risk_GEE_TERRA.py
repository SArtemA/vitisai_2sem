# fire_risk_GEE_TERRA.py
import ee
from datetime import datetime


def fire_risk_GEE_TERRA(
        input_data,
        lon=None,
        year: int = None,
        scale: int = 4000,
        verbose: bool = False,
        full_output: bool = False
):
    """
    Получает средний индекс засухи Палмера (PDSI) в качестве оценки пожарного риска (fire_risk)
    из датасета IDAHO_EPSCOR/TERRACLIMATE с автоматическим определением полушария.

    Суффикс ключа формируется на основе целевого года (например, _2024).

    Args:
        input_data: Либо ee.FeatureCollection (батч), либо широта (lat) если lon указан.
        lon: Долгота (передается только для одиночной точки).
        year: Год вегетационного сезона. Если None, используется (текущий_год - 1).
        scale: Пространственное разрешение для редукции (дефолт 4000 для TerraClimate).
        verbose: Флаг для детального логирования.
        full_output: Если True, возвращает сырой ответ от GEE.
    """
    # 1. Определение целевого года
    if year is None:
        year = datetime.now().year - 1

    if verbose:
        print(f"[INFO] Target vegetation year set to: {year}")

    # Базовое имя ключа и финальный ключ с суффиксом года
    base_key = 'fire_risk_mean_GEE_TERRA_'
    final_key = f"{base_key}{year}"

    try:
        # 2. Функция серверной обработки одной фичи
        def process_feature(feature):
            geom = feature.geometry()
            lat = ee.Number(geom.coordinates().get(1))

            # Временные интервалы для Северного и Южного полушарий
            start_north = ee.Date.fromYMD(year, 4, 1)
            end_north = ee.Date.fromYMD(year, 10, 31)

            start_south = ee.Date.fromYMD(year, 10, 1)
            end_south = ee.Date.fromYMD(year + 1, 4, 30)

            # Выбор дат на основе широты (сезон вегетации / пожароопасный период)
            start_date = ee.Date(ee.Algorithms.If(lat.gte(0), start_north, start_south))
            end_date = ee.Date(ee.Algorithms.If(lat.gte(0), end_north, end_south))

            # Фильтрация коллекции TerraClimate по датам и геометрии
            climate_col = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                           .filterDate(start_date, end_date.advance(1, 'day'))
                           .filterBounds(geom))

            # Масштабирование PDSI (исходный scale factor в TerraClimate = 0.1)
            def scale_pdsi(img):
                pdsi_scaled = img.select('pdsi').multiply(0.1).rename('pdsi')
                return img.addBands(pdsi_scaled, overwrite=True)

            climate_col_scaled = climate_col.map(scale_pdsi)

            # Расчет среднего значения PDSI за сезон
            pdsi_mean_img = climate_col_scaled.select('pdsi').mean().rename('pdsi')

            # Редукция данных в точке
            dict_values = pdsi_mean_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=geom,
                scale=scale
            )

            # Формирование выходного словаря с динамическим ключом года
            mapped_dict = ee.Dictionary({
                f'fire_risk_mean_GEE_TERRA_{year}': dict_values.get('pdsi')
            })

            return feature.set('climate_results', mapped_dict).set('orig_id', feature.id())

        # 3. Полиморфизм входных данных (Одиночная точка vs Пакетный режим)
        is_batch = isinstance(input_data, ee.FeatureCollection)

        if not is_batch:
            if verbose:
                print(f"[INFO] Mode: Single Point (Lat: {input_data}, Lon: {lon})")
            single_feature = ee.Feature(ee.Geometry.Point([lon, input_data]))
            features_to_process = ee.FeatureCollection([single_feature])
        else:
            if verbose:
                print(f"[INFO] Mode: Batch FeatureCollection (Size: {input_data.size().getInfo()})")
            features_to_process = input_data

        # Запуск обработки на сервере GEE
        processed_coll = features_to_process.map(process_feature)

        if full_output:
            return processed_coll.getInfo()

        if verbose:
            print("[INFO] Fetching data from Google Earth Engine servers...")

        client_results = processed_coll.select(['climate_results', 'orig_id']).getInfo()['features']

        # 4. Логика выравнивания и заполнения пропусков
        if not is_batch:
            res_dict = client_results[0]['properties']['climate_results']
            if res_dict.get(final_key) is None:
                return {final_key: 'error'}
            return res_dict
        else:
            output_list = []
            results_map = {}
            for feat in client_results:
                props = feat['properties']
                results_map[props['orig_id']] = props['climate_results']

            orig_ids = features_to_process.aggregate_array('system:index').getInfo()

            for orig_id in orig_ids:
                data = results_map.get(orig_id)
                if not data or data.get(final_key) is None:
                    output_list.append({final_key: 'error'})
                else:
                    output_list.append(data)

            if verbose:
                print(f"[SUCCESS] Successfully processed {len(output_list)} items.")
            return output_list

    except Exception as e:
        if verbose:
            print(f"[ERROR] GEE Fire Risk processing failed: {e}")
        if 'is_batch' in locals() and is_batch:
            try:
                size = input_data.size().getInfo()
                return [{final_key: 'error'} for _ in range(size)]
            except:
                return [{final_key: 'error'}]
        else:
            return {final_key: 'error'}


# ТЕСТОВЫЙ БЛОК ДЛЯ ПРОВЕРКИ РАБОТОСПОСОБНОСТИ
if __name__ == "__main__":
    import os
    import pprint
    from dotenv import load_dotenv

    print("--- Инициализация Google Earth Engine ---")
    load_dotenv()
    try:
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
        print("[SUCCESS] GEE успешно инициализирован.\n")
    except Exception as e:
        print("[ERROR] Не удалось инициализировать GEE. Требуется GEE_PROJECT_ID в .env")
        print(f"Детали ошибки: {e}")
        ee.Authenticate()
        ee.Initialize()

    TARGET_YEAR = 2024

    # -----------------------------------------------------------------
    # ТЕСТ 1: Одиночная точка
    # -----------------------------------------------------------------
    print("=" * 60)
    print("ТЕСТ 1: Одиночная корректная точка")
    print("=" * 60)

    result_single = fire_risk_GEE_TERRA(
        input_data=44.5172,
        lon=34.1844,
        year=TARGET_YEAR,
        scale=4000,
        verbose=True
    )
    print("\nРезультат Теста 1:")
    pprint.pprint(result_single)
    print("-" * 60)

    # -----------------------------------------------------------------
    # ТЕСТ 2: Пакетный режим ee.FeatureCollection
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Пакетный режим ee.FeatureCollection (2 точки)")
    print("=" * 60)

    pending_items = [
        (111111, 45.0123, 33.9876),    # Крым
        (222222, -33.8688, 151.2093),   # Австралия
        (333333, -0, 0)
    ]

    coords = [list(item[1:]) for item in pending_items]
    features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords])

    result_batch = fire_risk_GEE_TERRA(
        input_data=features,
        year=TARGET_YEAR,
        scale=4000,
        verbose=True
    )

    print("\nРезультат Теста 2:")
    for i, item_res in enumerate(result_batch):
        print(f"Элемент {pending_items[i][0]}: {item_res}")
    print("=" * 60)
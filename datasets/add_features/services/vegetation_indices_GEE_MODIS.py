# vegetation_indices_GEE_MODIS.py
import ee
from datetime import datetime


def vegetation_indices_GEE_MODIS(
        input_data,
        lon=None,
        year: int = None,
        scale: int = 250,
        verbose: bool = False,
        full_output: bool = False
):
    """
    Получает средние за сезон вегетационные индексы EVI и LAI из коллекций MODIS.
    Логика LAI оптимизирована пространственным сглаживанием, при этом строго
    сохраняется возвращение 'error' для точек без покрытия (океан, невалидные координаты).

    Args:
        input_data: Либо ee.FeatureCollection (батч), либо широта (lat) если lon указан.
        lon: Долгота (передается только для одиночной точки).
        year: Год вегетационного сезона. Если None, используется (текущий_год - 1).
        scale: Пространственное разрешение для редукции.
        verbose: Флаг для детального логирования.
        full_output: Если True, возвращает сырой ответ от GEE.
    """
    # 1. Определение целевого года
    if year is None:
        year = datetime.now().year - 1

    if verbose:
        print(f"[INFO] Target vegetation year set to: {year}")

    base_keys = [
        'evi_mean_GEE_MODIS_',
        'lai_mean_GEE_MODIS_'
    ]
    final_keys = [f"{key}{year}" for key in base_keys]

    try:
        # 2. Функция серверной обработки одной фичи
        def process_feature(feature):
            geom = feature.geometry()
            lat = ee.Number(geom.coordinates().get(1))

            # Временные интервалы для Северного и Южного полушарий
            start_north = ee.Date.fromYMD(year, 3, 1)
            end_north = ee.Date.fromYMD(year, 11, 30)

            start_south = ee.Date.fromYMD(year, 9, 1)
            end_south = ee.Date.fromYMD(year + 1, 5, 31)

            start_date = ee.Date(ee.Algorithms.If(lat.gte(0), start_north, start_south))
            end_date = ee.Date(ee.Algorithms.If(lat.gte(0), end_north, end_south))

            # --- СБОР EVI (MOD13Q1) ---
            evi_col = (ee.ImageCollection("MODIS/061/MOD13Q1")
                       .filterDate(start_date, end_date.advance(1, 'day'))
                       .filterBounds(geom))

            def scale_evi(img):
                scaled = img.select('EVI').multiply(0.0001).rename('evi')
                return img.addBands(scaled, overwrite=True)

            evi_mean_img = evi_col.map(scale_evi).select('evi').mean()

            # --- СБОР LAI (MCD15A3H) С АЛГОРИТМОМ ЗАЩИТЫ ОТ NONE ---
            lai_col = (ee.ImageCollection("MODIS/061/MCD15A3H")
                       .filterDate(start_date, end_date.advance(1, 'day'))
                       .filterBounds(geom))

            def scale_lai(img):
                scaled = img.select('Lai').multiply(0.1).rename('lai')
                return img.addBands(scaled, overwrite=True)

            lai_scaled_col = lai_col.map(scale_lai).select('lai')

            # Шаг А: Базовые временные композиты (Медиана и Максимум за сезон)
            lai_median = lai_scaled_col.median()
            lai_max = lai_scaled_col.max()

            # Шаг Б: Пространственное восстановление (Заполнение дыр за счет соседей в радиусе 1000м)
            # Внимание: unmask() убран из глобального растра, чтобы не занулять океан!
            lai_spatial_fill = lai_median.focalMean(radius=1000, units='meters')

            # Каскадное наложение только реальных спутниковых пикселей
            lai_final_img = lai_median.unmask(lai_max).unmask(lai_spatial_fill)

            # Компоновка растров
            final_metrics_img = ee.Image.cat([
                evi_mean_img.rename('evi'),
                lai_final_img.rename('lai')
            ])

            # Извлечение данных в точке
            dict_values = final_metrics_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=geom,
                scale=scale
            )

            # Серверный маппинг. Если в точке вообще нет покрытия MODIS (океан),
            # reduceRegion вернет словарь, где ключи 'evi' и 'lai' будут отсутствовать или равны null.
            mapped_dict = ee.Dictionary({
                f'evi_mean_GEE_MODIS_{year}': dict_values.get('evi'),
                f'lai_mean_GEE_MODIS_{year}': dict_values.get('lai')
            })

            return feature.set('modis_results', mapped_dict).set('orig_id', feature.id())

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
            print("[INFO] Fetching MODIS data from Google Earth Engine servers...")

        client_results = processed_coll.select(['modis_results', 'orig_id']).getInfo()['features']

        # 4. Логика строгого контроля ошибок (Клиентская часть)
        if not is_batch:
            res_dict = client_results[0]['properties']['modis_results']
            # Если хотя бы один ключевой признак отсутствует или равен None -> это критическая ошибка
            if res_dict.get(final_keys[0]) is None or res_dict.get(final_keys[1]) is None:
                return {k: 'error' for k in final_keys}
            return res_dict
        else:
            output_list = []
            results_map = {}
            for feat in client_results:
                props = feat['properties']
                results_map[props['orig_id']] = props['modis_results']

            orig_ids = features_to_process.aggregate_array('system:index').getInfo()

            for orig_id in orig_ids:
                data = results_map.get(orig_id)
                # Проверяем структуру и валидность извлеченных данных
                if not data or data.get(final_keys[0]) is None or data.get(final_keys[1]) is None:
                    output_list.append({k: 'error' for k in final_keys})
                else:
                    output_list.append(data)

            if verbose:
                print(f"[SUCCESS] Successfully processed {len(output_list)} items.")
            return output_list

    except Exception as e:
        if verbose:
            print(f"[ERROR] GEE MODIS processing failed: {e}")
        if 'is_batch' in locals() and is_batch:
            try:
                size = input_data.size().getInfo()
                return [{k: 'error' for k in final_keys} for _ in range(size)]
            except:
                return [{k: 'error' for k in final_keys}]
        else:
            return {k: 'error' for k in final_keys}


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
    # ТЕСТ 1: Одиночная точка (Крым)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("ТЕСТ 1: Одиночная корректная точка")
    print("=" * 60)

    result_single = vegetation_indices_GEE_MODIS(
        input_data=44.5172,
        lon=34.1844,
        year=TARGET_YEAR,
        scale=250,
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
        (111111, 45.0123, 33.9876),
        (222222, -33.8688, 151.2093),
        (333333, 0, 0)
    ]

    coords = [list(item[1:]) for item in pending_items]
    features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords])

    result_batch = vegetation_indices_GEE_MODIS(
        input_data=features,
        year=TARGET_YEAR,
        scale=250,
        verbose=True
    )

    print("\nРезультат Теста 2:")
    for i, item_res in enumerate(result_batch):
        print(f"Элемент {pending_items[i][0]}: {item_res}")
    print("=" * 60)
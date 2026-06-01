# landsat_GEE.py
import ee
from datetime import datetime

def landsat_GEE(
    input_data,
    lon=None,
    year: int = None,
    scale: int = 30,
    verbose: bool = False,
    full_output: bool = False
):
    """
    Получение данных Landsat 8 (спектральные каналы, индексы, температура)
    с автоматическим определением полушария и поддержкой батч-обработки.
    """
    # 1. Определение целевого года
    if year is None:
        year = datetime.now().year - 1

    suffix = f"_{year}"

    # Шаблон выходных данных на случай полной ошибки
    error_dict = {
        f'SR_B2_mean{suffix}': 'error',
        f'SR_B3_mean{suffix}': 'error',
        f'SR_B4_mean{suffix}': 'error',
        f'SR_B5_mean{suffix}': 'error',
        f'SR_B6_mean{suffix}': 'error',
        f'SR_B7_mean{suffix}': 'error',
        f'NDVI_phase1{suffix}': 'error',
        f'NDVI_phase2{suffix}': 'error',
        f'NDVI_phase3{suffix}': 'error',
        f'SAVI_phase2{suffix}': 'error',
        f'NDWI_phase2{suffix}': 'error',
        f'ST_B10_mean{suffix}': 'error',
        f'cloud_cover_phase2{suffix}': 'error'
    }

    # 3. Функция масштабирования каналов Landsat 8 (C02/T1_L2)
    def scale_landsat_bands(image):
        # Оптические каналы (Surface Reflectance)
        ops = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
                   .multiply(0.0000275).add(-0.2)
        # Тепловой канал (Surface Temperature)
        st = image.select('ST_B10').multiply(0.00341802).add(149.0)
        return image.addBands(ops, overwrite=True).addBands(st, overwrite=True)

    # 4. Функция маскирования облаков
    def mask_clouds(image):
        qa = image.select('QA_PIXEL')
        # Биты: 3 - Облако, 4 - Тень от облака, 5 - Снег/Лед
        cloud_mask = qa.bitwiseAnd(8).eq(0) \
            .And(qa.bitwiseAnd(16).eq(0)) \
            .And(qa.bitwiseAnd(32).eq(0))
        return image.updateMask(cloud_mask)

    # 5. Функция расчета индексов
    def add_indices(image):
        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

        # SAVI = ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
                'NIR': image.select('SR_B5'),
                'RED': image.select('SR_B4')
            }
        ).rename('SAVI')

        # NDWI = (NIR - SWIR1) / (NIR + SWIR1)
        ndwi = image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDWI')

        return image.addBands([ndvi, savi, ndwi])

    # Вспомогательная функция агрегации данных внутри GEE для конкретной геометрии и дат
    def process_phase_collection(geom, start_date, end_date, phase_num):
        coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                 .filterBounds(geom) \
                 .filterDate(start_date, end_date)

        # Применяем маску облаков к коллекции
        masked_coll = coll.map(mask_clouds)

        # Проверяем, есть ли снимки (размер отфильтрованной коллекции)
        has_images = masked_coll.size().gt(0)

        # Медианное значение по вегетационному периоду
        median_img = masked_coll.median()
        median_img = scale_landsat_bands(median_img)
        median_img = add_indices(median_img)

        # Извлекаем значения для точки
        reduced = median_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e9
        )
        
        # Формируем структуру ответа в зависимости от того, пуста ли коллекция чистых снимков
        return ee.Dictionary(ee.Algorithms.If(
            has_images,
            ee.Dictionary({
                'status': 'success',
                'SR_B2': reduced.get('SR_B2'),
                'SR_B3': reduced.get('SR_B3'),
                'SR_B4': reduced.get('SR_B4'),
                'SR_B5': reduced.get('SR_B5'),
                'SR_B6': reduced.get('SR_B6'),
                'SR_B7': reduced.get('SR_B7'),
                'ST_B10': reduced.get('ST_B10'),
                'NDVI': reduced.get('NDVI'),
                'SAVI': reduced.get('SAVI'),
                'NDWI': reduced.get('NDWI'),
                'cloud_cover': 0
            }),
            ee.Dictionary({
                'status': 'error',
                'cloud_cover': 1
            })
        ))

    # 7. Основная функция серверной обработки одной фичи (ee.Feature)
    def process_feature(feature):
        geom = feature.geometry()
        lat = geom.coordinates().get(1)
        
        is_northern = ee.Number(lat).gte(0)
        
        # Динамическое определение дат в зависимости от полушария
        # Северное полушарие
        n_p1_start, n_p1_end = f"{year}-04-01", f"{year}-05-31"
        n_p2_start, n_p2_end = f"{year}-07-01", f"{year}-08-31"
        n_p3_start, n_p3_end = f"{year}-09-01", f"{year}-10-31"
        
        # Южное полушарие (с переходом через границу года)
        s_p1_start, s_p1_end = f"{year}-10-01", f"{year}-11-30"
        s_p2_start, s_p2_end = f"{year+1}-01-01", f"{year+1}-02-28"
        s_p3_start, s_p3_end = f"{year+1}-03-01", f"{year+1}-04-30"
        
        p1_start = ee.Algorithms.If(is_northern, n_p1_start, s_p1_start)
        p1_end = ee.Algorithms.If(is_northern, n_p1_end, s_p1_end)
        
        p2_start = ee.Algorithms.If(is_northern, n_p2_start, s_p2_start)
        p2_end = ee.Algorithms.If(is_northern, n_p2_end, s_p2_end)
        
        p3_start = ee.Algorithms.If(is_northern, n_p3_start, s_p3_start)
        p3_end = ee.Algorithms.If(is_northern, n_p3_end, s_p3_end)
        
        # Получаем данные по всем фазам
        p1_data = process_phase_collection(geom, p1_start, p1_end, 1)
        p2_data = process_phase_collection(geom, p2_start, p2_end, 2)
        p3_data = process_phase_collection(geom, p3_start, p3_end, 3)
        
        # Объединяем результаты с проверкой статусов на сервере GEE
        def get_val(phase_data, key, default_val='error'):
            return ee.Algorithms.If(
                ee.String(phase_data.get('status')).equals('success'),
                ee.Algorithms.If(ee.Algorithms.IsEqual(phase_data.get(key), None), 'error', phase_data.get(key)),
                default_val
            )

        res_dict = ee.Dictionary({
            # Спектральные каналы и температура берутся только из Phase 2 (пик сезона)
            f'SR_B2_mean{suffix}': get_val(p2_data, 'SR_B2'),
            f'SR_B3_mean{suffix}': get_val(p2_data, 'SR_B3'),
            f'SR_B4_mean{suffix}': get_val(p2_data, 'SR_B4'),
            f'SR_B5_mean{suffix}': get_val(p2_data, 'SR_B5'),
            f'SR_B6_mean{suffix}': get_val(p2_data, 'SR_B6'),
            f'SR_B7_mean{suffix}': get_val(p2_data, 'SR_B7'),
            f'ST_B10_mean{suffix}': get_val(p2_data, 'ST_B10'),
            
            # Вегетационные индексы по фазам
            f'NDVI_phase1{suffix}': get_val(p1_data, 'NDVI'),
            f'NDVI_phase2{suffix}': get_val(p2_data, 'NDVI'),
            f'NDVI_phase3{suffix}': get_val(p3_data, 'NDVI'),
            
            # SAVI и NDWI только для пика сезона
            f'SAVI_phase2{suffix}': get_val(p2_data, 'SAVI'),
            f'NDWI_phase2{suffix}': get_val(p2_data, 'NDWI'),
            
            # Облачность (если phase2 заблокирован облаками, выставит 1, иначе 0)
            f'cloud_cover_phase2{suffix}': p2_data.get('cloud_cover')
        })
        
        # Сохраняем исходный системный индекс для сортировки в батч-режиме
        return feature.set('landsat_outputs', res_dict).set('orig_index', feature.get('system:index'))

    # 8. Полиморфизм входных данных (одиночная точка vs FeatureCollection)
    try:
        if isinstance(input_data, (int, float)):
            # Одиночная точка: input_data — это latitude
            lat = float(input_data)
            lon = float(lon)
            single_feature = ee.Feature(ee.Geometry.Point([lon, lat]))
            processed_feature = process_feature(single_feature)
            
            # Запрос к серверу GEE
            raw_result = processed_feature.get('landsat_outputs').getInfo()
            if full_output:
                return raw_result
            return raw_result

        elif isinstance(input_data, ee.FeatureCollection):
            # Батч-обработка коллекции
            if verbose:
                print(f"Запуск батч-обработки Landsat 8 для {input_data.size().getInfo()} объектов...")
            
            # Применяем маппинг на сервере GEE
            processed_coll = input_data.map(process_feature)
            
            # Извлекаем данные, сортируя по исходному порядку system:index
            features_list = processed_coll.sort('orig_index').getInfo()['features']
            
            results = []
            for feat in features_list:
                outputs = feat['properties'].get('landsat_outputs', error_dict)
                results.append(outputs)
                
            return results
        else:
            if verbose:
                print("Неизвестный тип входных данных input_data.")
            return error_dict if not isinstance(input_data, ee.FeatureCollection) else [error_dict]

    except Exception as e:
        if verbose:
            print(f"Ошибка при работе с Google Earth Engine: {e}")
        if isinstance(input_data, ee.FeatureCollection):
            # Возвращаем список заглушек в случае падения сетевого запроса
            try:
                count = input_data.size().getInfo()
                return [error_dict] * count
            except:
                return [error_dict]
        return error_dict


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    print("--- Инициализация Google Earth Engine ---")
    load_dotenv()
    try:
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))
        print("[SUCCESS] GEE успешно инициализирован.\n")
    except Exception as e:
        print("[ERROR] Не удалось инициализировать GEE. Требуется авторизация или GEE_PROJECT_ID в .env")
        print(f"Детали ошибки: {e}")
        ee.Authenticate()
        ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))

    TARGET_YEAR = 2024

    print("=" * 60)
    print("ТЕСТ 1: Одиночная точка на суше (Крым, Северное полушарие)")
    print("=" * 60)

    try:
        result_single_land = landsat_GEE(
            input_data=45.0123,
            lon=33.9876,
            year=TARGET_YEAR,
            scale=30,
            verbose=True,
            full_output=False
        )

        print(f"\nРезультат для точки (45.0123, 33.9876):")
        print(f"NDVI_phase2_{TARGET_YEAR}: {result_single_land.get(f'NDVI_phase2_{TARGET_YEAR}')}")
        print(f"ST_B10_mean_{TARGET_YEAR}: {result_single_land.get(f'ST_B10_mean_{TARGET_YEAR}')}")
        print(f"cloud_cover_phase2_{TARGET_YEAR}: {result_single_land.get(f'cloud_cover_phase2_{TARGET_YEAR}')}")

        print(f"Вид dict:\n{result_single_land}")

    except Exception as e:
        print(f"ОШИБКА в тесте 1: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТ 2: Одиночная точка в океане (должна вернуть error)")
    print("=" * 60)

    try:
        result_single_ocean = landsat_GEE(
            input_data=0.0,
            lon=0.0,
            year=TARGET_YEAR,
            scale=30,
            verbose=True,
            full_output=False
        )

        print(f"\nРезультат для точки в океане (0.0, 0.0):")
        ndvi_value = result_single_ocean.get(f'NDVI_phase2_{TARGET_YEAR}')
        cloud_cover = result_single_ocean.get(f'cloud_cover_phase2_{TARGET_YEAR}')
        print(f"NDVI_phase2_{TARGET_YEAR}: {ndvi_value}")
        print(f"cloud_cover_phase2_{TARGET_YEAR}: {cloud_cover}")

    except Exception as e:
        print(f"ОШИБКА в тесте 2: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТ 3: Батч-обработка 3 точек (Крым, Океан, Австралия)")
    print("=" * 60)

    pending_items = [
        (111111, 45.0123, 33.9876),
        (222222, 0.0, 0.0),
        (333333, -33.8688, 151.2093)
    ]

    features_list = []
    for osm_id, lat, lon in pending_items:
        feature = ee.Feature(ee.Geometry.Point([lon, lat]))
        feature = feature.set('osm_id', osm_id)
        features_list.append(feature)

    feature_collection = ee.FeatureCollection(features_list)

    print(f"Создана FeatureCollection из {len(pending_items)} объектов")

    try:
        batch_results = landsat_GEE(
            input_data=feature_collection,
            year=TARGET_YEAR,
            scale=30,
            verbose=True,
            full_output=False
        )

        print(f"\nПолучено {len(batch_results)} результатов:")

        for i, (item, result) in enumerate(zip(pending_items, batch_results)):
            osm_id, lat, lon = item
            print(f"\nОбъект {i + 1} (OSM ID: {osm_id}, {lat}, {lon})")

            ndvi = result.get(f'NDVI_phase2_{TARGET_YEAR}')
            temp = result.get(f'ST_B10_mean_{TARGET_YEAR}')
            cloud = result.get(f'cloud_cover_phase2_{TARGET_YEAR}')
            savi = result.get(f'SAVI_phase2_{TARGET_YEAR}')
            ndwi = result.get(f'NDWI_phase2_{TARGET_YEAR}')

            print(f"  NDVI_phase2: {ndvi}")
            print(f"  SAVI_phase2: {savi}")
            print(f"  NDWI_phase2: {ndwi}")
            print(f"  ST_B10_mean: {temp}")
            print(f"  cloud_cover: {cloud}")

        success_count = sum(1 for r in batch_results if r.get(f'NDVI_phase2_{TARGET_YEAR}') != 'error')
        error_count = len(batch_results) - success_count
        print(f"\nУспешных точек: {success_count}")
        print(f"Точек с ошибкой: {error_count}")

    except Exception as e:
        print(f"ОШИБКА в тесте 3: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("ТЕСТ 4: Некорректные координаты")
    print("=" * 60)

    try:
        result_invalid = landsat_GEE(
            input_data=999.0,
            lon=999.0,
            year=TARGET_YEAR,
            scale=30,
            verbose=True,
            full_output=False
        )

        print(f"\nРезультат для невалидных координат (999.0, 999.0):")
        ndvi_value = result_invalid.get(f'NDVI_phase2_{TARGET_YEAR}')
        print(f"NDVI_phase2_{TARGET_YEAR}: {ndvi_value}")

    except Exception as e:
        print(f"Исключение при невалидных координатах: {e}")

    print("\n" + "=" * 60)
    print("ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
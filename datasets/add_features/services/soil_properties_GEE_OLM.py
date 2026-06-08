# soil_properties_GEE_OLM.py
import ee


def soil_properties_GEE_OLM(
        input_data,
        lon=None,
        scale: int = 30,
        verbose: bool = False,
        full_output: bool = False
):
    """
    Получает статические свойства почвы (pH и органический углерод) для верхнего слоя (0 см)
    из датасетов OpenLandMap с пространственным разрешением 30 метров.

    Поскольку данные статические, суффикс года в ключах не используется.

    Args:
        input_data: Либо ee.FeatureCollection (батч), либо широта (lat) если lon указан.
        lon: Долгота (передается только для одиночной точки).
        scale: Пространственное разрешение для редукции (30 метров, как в спецификации OpenLandMap).
        verbose: Флаг для детального логирования.
        full_output: Если True, возвращает сырой ответ от GEE.
    """
    # Базовые имена ключей в выходном словаре (без привязки к году)
    final_keys = [
        'soil_ph_GEE_OLM',
        'soil_organic_carbon_GEE_OLM'
    ]

    try:
        # 1. Функция серверной обработки одной фичи
        def process_feature(feature):
            geom = feature.geometry()

            # Загружаем растры OpenLandMap
            # b0 — это слой, соответствующий глубине 0 см (поверхность почвы)
            ph_img = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02").select('b0')
            soc_img = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0')

            # Масштабирование pH: исходные данные хранятся как pH*10, делим на 10
            ph_scaled = ph_img.divide(10.0).rename('soil_ph')

            # Органический углерод (g/kg) берем как есть
            soc_scaled = soc_img.rename('soil_organic_carbon')

            # Компонуем растры в один
            final_metrics_img = ee.Image.cat([ph_scaled, soc_scaled])

            # Редукция региона в точке
            dict_values = final_metrics_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=geom,
                scale=scale
            )

            # Мэппинг в итоговые ключи
            mapped_dict = ee.Dictionary({
                'soil_ph_GEE_OLM': dict_values.get('soil_ph'),
                'soil_organic_carbon_GEE_OLM': dict_values.get('soil_organic_carbon')
            })

            return feature.set('soil_results', mapped_dict).set('orig_id', feature.id())

        # 2. Полиморфизм входных данных (Одиночная точка vs Пакетный режим)
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
            print("[INFO] Fetching soil data from Google Earth Engine servers...")

        client_results = processed_coll.select(['soil_results', 'orig_id']).getInfo()['features']

        # 3. Логика выравнивания и заполнения пропусков
        if not is_batch:
            res_dict = client_results[0]['properties']['soil_results']
            if res_dict.get(final_keys[0]) is None:
                return {k: 'error' for k in final_keys}
            return res_dict
        else:
            output_list = []
            results_map = {}
            for feat in client_results:
                props = feat['properties']
                results_map[props['orig_id']] = props['soil_results']

            orig_ids = features_to_process.aggregate_array('system:index').getInfo()

            for orig_id in orig_ids:
                data = results_map.get(orig_id)
                if not data or data.get(final_keys[0]) is None:
                    output_list.append({k: 'error' for k in final_keys})
                else:
                    output_list.append(data)

            if verbose:
                print(f"[SUCCESS] Successfully processed {len(output_list)} items.")
            return output_list

    except Exception as e:
        if verbose:
            print(f"[ERROR] GEE Soil properties processing failed: {e}")
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

    # -----------------------------------------------------------------
    # ТЕСТ 1: Одиночная точка (Крым, Массандра)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("ТЕСТ 1: Одиночная корректная точка")
    print("=" * 60)

    lat_valid = 44.5172
    lon_valid = 34.1844

    result_single = soil_properties_GEE_OLM(
        input_data=lat_valid,
        lon=lon_valid,
        scale=30,
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

    result_batch = soil_properties_GEE_OLM(
        input_data=features,
        scale=30,
        verbose=True
    )

    print("\nРезультат Теста 2:")
    for i, item_res in enumerate(result_batch):
        print(f"Элемент {pending_items[i][0]}: {item_res}")
    print("=" * 60)
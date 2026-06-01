# climate_GEE_TERRA.py
import ee
from datetime import datetime


def climate_GEE_TERRA(
        input_data,
        lon=None,
        year: int = None,
        scale: int = 4000,
        verbose: bool = False,
        full_output: bool = False
):
    """
    Получает сезонные климатические данные (температуры, осадки, дефицит влаги, радиацию, GDD)
    из датасета IDAHO_EPSCOR/TERRACLIMATE с автоматическим определением полушария для каждой точки.

    Суффикс ключей формируется на основе целевого года (например, _2024).

    Args:
        input_data: Либо ee.FeatureCollection (батч), либо широта (lat) если lon указан.
        lon: Долгота (передается только для одиночной точки).
        year: Год вегетационного сезона. Если None, используется (текущий_год - 1).
        scale: Пространственное разрешение для редукции (для TerraClimate дефолт 4000).
        verbose: Флаг для детального логирования.
        full_output: Если True, возвращает сырой ответ от GEE.
    """
    # 1. Определение целевого года
    if year is None:
        year = datetime.now().year - 1

    if verbose:
        print(f"[INFO] Target vegetation year set to: {year}")

    # Базовые имена ключей (суффикс _ГОД добавится динамически)
    base_keys = [
        'tmax_mean_GEE_TERRA_',
        'tmin_mean_GEE_TERRA_',
        'winkler_gdd_total_GEE_TERRA_',
        'precip_total_GEE_TERRA_',
        'water_deficit_total_GEE_TERRA_',
        'solar_rad_mean_GEE_TERRA_'
    ]

    # Итоговые имена ключей в выходном словаре
    final_keys = [f"{key}{year}" for key in base_keys]

    try:
        # 2. Функция серверной обработки одной фичи (работает и в батче, и для одиночной точки)
        def process_feature(feature):
            geom = feature.geometry()
            lat = ee.Number(geom.coordinates().get(1))

            # Временные интервалы для Северного и Южного полушарий
            start_north = ee.Date.fromYMD(year, 4, 1)
            end_north = ee.Date.fromYMD(year, 10, 31)

            start_south = ee.Date.fromYMD(year, 10, 1)
            end_south = ee.Date.fromYMD(year + 1, 4, 30)

            # Условие GEE для выбора дат и количества дней вегетации на основе широты
            start_date = ee.Date(ee.Algorithms.If(lat.gte(0), start_north, start_south))
            end_date = ee.Date(ee.Algorithms.If(lat.gte(0), end_north, end_south))

            # Количество дней в вегетационном периоде (214 для С.П., 212 для Ю.П.)
            days_in_season = ee.Number(ee.Algorithms.If(lat.gte(0), 214, 212))

            # Фильтрация коллекции TerraClimate по датам и геометрии
            climate_col = (ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
                           .filterDate(start_date, end_date.advance(1, 'day'))
                           .filterBounds(geom))

            # Перевод температур из дециградусов в градусы Цельсия (scale factor = 0.1)
            # Заменены каналы: tmax -> tmmx, tmin -> tmmn
            def scale_temperatures(img):
                tmax_scaled = img.select('tmmx').multiply(0.1).rename('tmax')
                tmin_scaled = img.select('tmmn').multiply(0.1).rename('tmin')
                return img.addBands([tmax_scaled, tmin_scaled], overwrite=True)

            climate_col_scaled = climate_col.map(scale_temperatures)

            # Расчет средних температур для Winkler GDD
            tmax_mean_img = climate_col_scaled.select('tmax').mean()
            tmin_mean_img = climate_col_scaled.select('tmin').mean()

            # Расчет Winkler GDD на основе средних за сезон температур и количества дней вегетации
            gdd_base = tmax_mean_img.add(tmin_mean_img).divide(2).subtract(10)
            gdd_img = gdd_base.max(0).multiply(days_in_season).rename('gdd')

            # Агрегация остальных параметров
            precip_sum_img = climate_col_scaled.select('pr').sum().rename('precip')
            vpd_sum_img = climate_col_scaled.select('vpd').sum().rename('vpd')
            srad_mean_img = climate_col_scaled.select('srad').mean().rename('srad')

            # Компоновка всех рассчитанных метрик в один растр
            final_metrics_img = ee.Image.cat([
                tmax_mean_img.rename('tmax'),
                tmin_mean_img.rename('tmin'),
                gdd_img,
                precip_sum_img,
                vpd_sum_img,
                srad_mean_img
            ])

            # Редукция региона (извлечение данных в точке)
            dict_values = final_metrics_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=geom,
                scale=scale
            )

            # Мэппинг оригинальных каналов GEE в требуемые ключи с суффиксом года
            mapped_dict = ee.Dictionary({
                f'tmax_mean_GEE_TERRA_{year}': dict_values.get('tmax'),
                f'tmin_mean_GEE_TERRA_{year}': dict_values.get('tmin'),
                f'winkler_gdd_total_GEE_TERRA_{year}': dict_values.get('gdd'),
                f'precip_total_GEE_TERRA_{year}': dict_values.get('precip'),
                f'water_deficit_total_GEE_TERRA_{year}': dict_values.get('vpd'),
                f'solar_rad_mean_GEE_TERRA_{year}': dict_values.get('srad')
            })

            # Сохранение системного индекса для последующей синхронизации пропусков
            return feature.set('climate_results', mapped_dict).set('orig_id', feature.id())

        # 3. Полиморфизм входных данных (Одиночная точка vs Пакетный режим)
        is_batch = isinstance(input_data, ee.FeatureCollection)

        if not is_batch:
            if verbose:
                print(f"[INFO] Mode: Single Point (Lat: {input_data}, Lon: {lon})")
            # Конвертируем одиночную точку в ee.FeatureCollection из одного элемента
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

        # Выгрузка результатов на сторону клиента
        if verbose:
            print("[INFO] Fetching data from Google Earth Engine servers...")

        client_results = processed_coll.select(['climate_results', 'orig_id']).getInfo()['features']

        # 4. Логика выравнивания и заполнения пропусков
        if not is_batch:
            # Для одиночной точки возвращаем плоский словарь
            res_dict = client_results[0]['properties']['climate_results']
            # Если точка пустая (нет покрытия), заполняем 'error'
            if res_dict.get(final_keys[0]) is None:
                return {k: 'error' for k in final_keys}
            return res_dict
        else:
            # Для батча синхронизируем данные по исходному порядку system:index
            output_list = []

            results_map = {}
            for feat in client_results:
                props = feat['properties']
                results_map[props['orig_id']] = props['climate_results']

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
            print(f"[ERROR] GEE Climate processing failed: {e}")
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
        ee.Initialize()

    TARGET_YEAR = 2024

    # -----------------------------------------------------------------
    # ТЕСТ 1: Точка выдаст показатели - одна точка (Крым, Массандра)
    # -----------------------------------------------------------------
    print("=" * 60)
    print("ТЕСТ 1: Одиночная корректная точка (Северное полушарие)")
    print("=" * 60)

    lat_valid = 44.5172
    lon_valid = 34.1844

    result_single = climate_GEE_TERRA(
        input_data=lat_valid,
        lon=lon_valid,
        year=TARGET_YEAR,
        verbose=True
    )
    print("\nРезультат Теста 1:")
    import pprint

    pprint.pprint(result_single)
    print("-" * 60)

    # -----------------------------------------------------------------
    # ТЕСТ 2: Точка выдаст error со стороны gee - одна точка (Некорректные координаты)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Одиночная точка за пределами Земли (Ожидается 'error')")
    print("=" * 60)

    lat_invalid = 999.0
    lon_invalid = 999.0

    result_error = climate_GEE_TERRA(
        input_data=lat_invalid,
        lon=lon_invalid,
        year=TARGET_YEAR,
        verbose=True
    )
    print("\nРезультат Теста 2:")
    pprint.pprint(result_error)
    print("-" * 60)

    # -----------------------------------------------------------------
    # ТЕСТ 3: Набор ee.FeatureCollection - внутри 3 точки
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Пакетный режим ee.FeatureCollection (3 точки)")
    print("=" * 60)

    # Структура данных: [osm_id, lat, lon]
    pending_items = [
        (111111, 45.0123, 33.9876),  # Точка 1: Крым
        (222222, 999.0, 999.0),  # Точка 2: Битый элемент (должен вернуть 'error' внутри списка)
        (333333, -33.8688, 151.2093)  # Точка 3: Австралия (Сидней, Южное полушарие)
    ]

    # Создание FeatureCollection
    coords = [list(item[1:]) for item in pending_items]
    features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords])

    print("[INFO] Отправка FeatureCollection в функцию...")
    result_batch = climate_GEE_TERRA(
        input_data=features,
        year=TARGET_YEAR,
        verbose=True
    )

    print("\nРезультат Теста 3 (Список словарей):")
    for i, item_res in enumerate(result_batch):
        print(f"\nИсходный элемент {pending_items[i]}:")
        pprint.pprint(item_res)
    print("=" * 60)
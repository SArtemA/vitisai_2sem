# init_db.py - Инициализация БД
import pandas as pd
import geopandas as gpd
from pathlib import Path

from numpy.ma.core import negative

from schema import setup_database
from repository import insert_points


def load_geojson_to_db(geojson_path, db_path, table_name: str):
    """
    Извлекает все позиции из geojson формата.
    В geojson требуются только 'osm_id' и 'geometry'
    """
    # Преобразование в Path объекты
    geojson_path = Path(geojson_path)
    db_path = Path(db_path)

    # Проверка существования GeoJson файла
    if not geojson_path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {geojson_path}")

    # Проверка существования базы данных
    if not db_path.exists():
        raise FileNotFoundError(f"База данных не найдена: {db_path}")

    gdf = gpd.read_file(geojson_path)

    required_attr = {"osm_id"}
    if not required_attr.issubset(gdf.columns):
        raise ValueError(f"GeoJSON должен содержать атрибут {required_attr}")

    # Извлекаем данные: id, широту и долготу из геометрии
    # geometry.y — это lat, geometry.x — это lon
    points = []
    for _, row in gdf.iterrows():
        geom = row["geometry"]
        if geom is None:
            continue

        # Если это точка — берем её координаты напрямую
        if geom.geom_type == 'Point':
            lat, lon = geom.y, geom.x
        # Если это полигон или линия — берем центр (centroid)
        else:
            centroid = geom.centroid
            lat, lon = centroid.y, centroid.x

        points.append((row["osm_id"], lat, lon))

    insert_points(points, db_path, table_name)


def load_csv_to_db(csv_file_path, db_path, table_name: str):
    """Загружает данные из CSV файла в указанную таблицу базы данных."""
    # Преобразование в Path объекты
    csv_path = Path(csv_file_path)
    db_path = Path(db_path)

    # Проверка существования CSV файла
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

    # Проверка существования базы данных
    if not db_path.exists():
        raise FileNotFoundError(f"База данных не найдена: {db_path}")

    try:
        # Чтение CSV файла с автоматическим определением разделителя
        df = pd.read_csv(csv_path)

        # Проверка наличия обязательных столбцов (регистронезависимо)
        required_columns = {'osm_id', 'lat', 'lon'}
        df_columns_lower = {col.lower(): col for col in df.columns}

        missing_columns = required_columns - set(df_columns_lower.keys())
        if missing_columns:
            raise ValueError(
                f"В CSV файле отсутствуют обязательные столбцы: {missing_columns}\n"
                f"Доступные столбцы: {list(df.columns)}"
            )

        # Переименовываем столбцы в стандартный вид (если нужно)
        rename_mapping = {df_columns_lower[col]: col for col in required_columns}
        df = df.rename(columns=rename_mapping)

        # Оставляем только нужные столбцы
        df = df[['osm_id', 'lat', 'lon']].copy()

        # Приведение типов данных
        df['osm_id'] = pd.to_numeric(df['osm_id'], errors='coerce')
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

        # Удаление строк с NaN значениями
        initial_count = len(df)
        df = df.dropna()
        nan_count = initial_count - len(df)
        if nan_count > 0:
            print(f"Предупреждение: Удалено {nan_count} строк с некорректными значениями")

        # Валидация координат
        valid_lat_mask = (df['lat'] >= -90) & (df['lat'] <= 90)
        valid_lon_mask = (df['lon'] >= -180) & (df['lon'] <= 180)

        invalid_count = (~(valid_lat_mask & valid_lon_mask)).sum()
        df = df[valid_lat_mask & valid_lon_mask]

        if invalid_count > 0:
            print(f"Предупреждение: Отфильтровано {invalid_count} строк с некорректными координатами")

        if len(df) == 0:
            print("Внимание: Не найдено корректных данных для загрузки")
            return 0

        # Преобразование в список кортежей для вставки
        points_data = list(df.itertuples(index=False, name=None))

        insert_points(points_data, db_path, table_name)

    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV файл пуст: {csv_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Ошибка парсинга CSV файла: {e}")


if __name__ == "__main__":
    # Путь к БД
    db_folder_path = Path(__file__).resolve().parent.parent / "data"
    db_folder_path.mkdir(exist_ok=True)
    db_name = 'vineyard_1.db'
    db_path = db_folder_path / db_name

    # Путь к Данным vineyard
    vineyard_path = Path(__file__).resolve().parent.parent / "data" / "merged.geojson"

    # Путь к Данным negative
    negative_path = Path(__file__).resolve().parent.parent / "data" / "output_negative.csv"

    print("Создание БД...")
    setup_database(db_path)

    print("="*30)
    print("Загрузка vineyard_features позиции\n",vineyard_path)
    # load_geojson_to_db(vineyard_path, db_path, "vineyard_features")

    print("Загрузка negative_features позиции\n",negative_path)
    load_csv_to_db(negative_path, db_path, "negative_features")

    print("Done")

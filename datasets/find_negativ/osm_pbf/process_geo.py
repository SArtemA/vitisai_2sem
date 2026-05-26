import csv
from pathlib import Path
import geopandas as gpd


def find_geojson_files(base_dir: Path) -> list[Path]:
    """Рекурсивно находит все файлы .geojson в указанной папке и подпапках."""
    return list(base_dir.rglob("*.geojson"))


def extract_and_filter_polygons(file_path: Path) -> gpd.GeoDataFrame | None:
    """Читает файл, фильтрует только Polygon, проверяет валидность и площадь."""
    try:
        # Читаем файл
        gdf = gpd.read_file(file_path)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path.name}: {e}")
        return None

    if gdf.empty:
        return None

    # 1. Отбираем только тип Polygon
    gdf = gdf[gdf.geometry.geom_type == "Polygon"]
    if gdf.empty:
        return None

    # 2. Проверка на валидность геометрии
    gdf = gdf[gdf.geometry.is_valid].copy()
    if gdf.empty:
        return None

    # Устанавливаем WGS 84, если она не задана
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # 3. Переводим в метровую проекцию для точного расчета площади (например, EPSG:3857)
    # Это уберет UserWarning и даст площадь в квадратных метрах
    gdf_metric = gdf.to_crs(epsg=3857)

    # Считаем площадь в кв. метрах и добавляем колонку в исходный WGS84 датафрейм
    gdf["area_sq_m"] = gdf_metric.geometry.area

    # 4. Фильтруем: оставляем только полигоны площадью больше 10 000 кв. м
    gdf = gdf[gdf["area_sq_m"] > 10000]
    if gdf.empty:
        return None

    # Ищем колонку с OSM ID
    id_col = None
    for col in ["osm_id", "id", "OSM_ID"]:
        if col in gdf.columns:
            id_col = col
            break

    if id_col:
        gdf = gdf[[id_col, "geometry", "area_sq_m"]].rename(columns={id_col: "osm_id"})
    else:
        gdf = gdf[["geometry", "area_sq_m"]].copy()
        gdf["osm_id"] = gdf.index

    return gdf


def save_centroids_to_csv(geo_data_list: list[gpd.GeoDataFrame], output_path: Path):
    """Вычисляет центроиды полигонов и сохраняет данные в один CSV-файл."""
    if not geo_data_list:
        print("Нет данных для сохранения.")
        return

    # Объединяем все датафреймы
    combined_gdf = gpd.GeoDataFrame(gpd.pd.concat(geo_data_list, ignore_index=True))

    print(f"Запись центроидов в {output_path.name}...")

    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Добавлен столбец area_sq_m
        writer.writerow(["osm_id", "lat", "lon", "area_sq_m"])

        for _, row in combined_gdf.iterrows():
            osm_id = row["osm_id"]
            polygon = row["geometry"]
            area_sq_m = row["area_sq_m"]

            # Вычисляем центроид полигона (в системе WGS 84)
            centroid = polygon.centroid

            # В WGS 84 x — это долгота (lon), y — широта (lat)
            lon = centroid.x
            lat = centroid.y

            # Округлим площадь для красоты до 2 знаков после запятой
            writer.writerow([osm_id, lat, lon, round(area_sq_m, 2)])


def main():
    current_dir = Path(__file__).parent
    geofiles_dir = current_dir / "geofiles"
    output_csv = current_dir / "output_polygons.csv"

    if not geofiles_dir.exists():
        print(f"Папка {geofiles_dir} не найдена!")
        return

    print("Поиск файлов .geojson...")
    geojson_files = find_geojson_files(geofiles_dir)
    print(f"Найдено файлов: {len(geojson_files)}")

    all_polygons = []
    for file_path in geojson_files:
        print(f"Обработка: {file_path.relative_to(current_dir)}")
        gdf = extract_and_filter_polygons(file_path)
        if gdf is not None and not gdf.empty:
            all_polygons.append(gdf)

    print(f"Успешно обработано файлов с полигонами: {len(all_polygons)}")

    save_centroids_to_csv(all_polygons, output_csv)
    print("Готово!")


if __name__ == "__main__":
    main()
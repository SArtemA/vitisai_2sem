import csv
import random
from pathlib import Path
import xml.etree.ElementTree as ET


def csv_to_kml_sampled(
    csv_filename: str, kml_filename: str, sample_percent: float, seed: int = None
):
    """Конвертирует CSV в KML с возможностью случайного отбора точек."""
    if seed is not None:
        random.seed(seed)

    sample_rate = sample_percent / 100.0

    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    csv_path = current_dir / csv_filename
    kml_path = current_dir / kml_filename

    if not csv_path.exists():
        print(f"Ошибка: Файл {csv_filename} не найден в папке {current_dir}")
        return

    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    document = ET.SubElement(kml, "Document")

    doc_name = ET.SubElement(document, "name")
    doc_name.text = f"Points Sample ({sample_percent}%)"

    total_rows = 0
    saved_points = 0

    with csv_path.open(mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            total_rows += 1

            if random.random() > sample_rate:
                continue

            try:
                osm_id = row["osm_id"]
                lon = row["lon"].strip()
                lat = row["lat"].strip()
            except KeyError as e:
                print(
                    f"Ошибка: В CSV файле не найдена колонка {e}. Проверьте заголовки."
                )
                return

            placemark = ET.SubElement(document, "Placemark")
            name = ET.SubElement(placemark, "name")
            name.text = f"OSM ID: {osm_id}"

            # Корректное создание XML-узлов для координат
            point = ET.SubElement(placemark, "Point")
            coordinates = ET.SubElement(point, "coordinates")
            coordinates.text = f"{lon},{lat}"

            saved_points += 1

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ", level=0)

    with kml_path.open(mode="wb") as kml_file:
        tree.write(kml_file, encoding="utf-8", xml_declaration=True)

    print(f"Обработано строк в CSV: {total_rows}")
    print(f"Успешно сохранено точек: {saved_points}")
    print(f"Файл сохранен как: {kml_path.name}")


# Запуск конвертера
if __name__ == "__main__":
    csv_to_kml_sampled(
        csv_filename="output_polygons.csv",
        kml_filename="output.kml",
        sample_percent=0.02,
        seed=1
    )
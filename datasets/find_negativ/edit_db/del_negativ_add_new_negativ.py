import csv
import sqlite3
from pathlib import Path


def process_vineyard_data():
    # Определение путей к файлам через pathlib
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    db_path = current_dir.parent.parent.parent / "vineyards_v2.db"
    csv_path = current_dir.parent / "osm_pbf" / "choose_farmland.csv"

    # Проверяем существование файлов
    if not db_path.exists():
        print(f"Ошибка: Файл базы данных не найден по пути {db_path}")
        return
    if not csv_path.exists():
        print(f"Ошибка: CSV-файл не найден по пути {csv_path}")
        return

    # Подключение к базе данных SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # 1. Удаление строк, где is_suitable равен 0
        cursor.execute(
            "DELETE FROM vineyard_features WHERE is_suitable = 0"
        )
        deleted_rows = cursor.rowcount
        print(f"Успешно удалено строк: {deleted_rows}")

        # 2. Чтение данных из CSV и подготовка к вставке
        insert_data = []

        with open(csv_path, mode="r", encoding="utf-8") as csv_file:
            # DictReader автоматически использует первую строку под заголовки
            reader = csv.DictReader(csv_file)

            for row in reader:
                # Извлекаем только нужные столбцы и добавляем 0 для is_suitable
                osm_id = row["osm_id"]
                lat = row["lat"]
                lon = row["lon"]
                is_suitable = 0

                insert_data.append((osm_id, lat, lon, is_suitable))

        # 3. Добавление новых позиций в таблицу
        # Названия колонок (osm_id, lat, lon) могут отличаться в вашей БД,
        # при необходимости скорректируйте их внутри скобок перед VALUES
        cursor.executemany(
            """
            INSERT INTO vineyard_features (osm_id, lat, lon, is_suitable)
            VALUES (?, ?, ?, ?)
        """,
            insert_data,
        )

        inserted_rows = cursor.rowcount
        print(f"Успешно добавлено новых строк из CSV: {inserted_rows}")

        # Сохраняем изменения в базе данных
        conn.commit()
        print("Все изменения успешно сохранены.")

    except sqlite3.Error as e:
        # Если произошла ошибка, откатываем изменения
        conn.rollback()
        print(f"Произошла ошибка при работе с БД: {e}")

    except KeyError as e:
        print(f"Ошибка: В CSV-файле отсутствует необходимый столбец {e}")

    finally:
        # Обязательно закрываем соединение
        conn.close()


if __name__ == "__main__":
    process_vineyard_data()
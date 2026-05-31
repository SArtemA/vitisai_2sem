import sqlite3
from pathlib import Path


def setup_database(db_path):
    """Создаёт БД с двумя таблицами: vineyard_features и negative_features."""
    try:
        # Подключение к БД
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Включение поддержки внешних ключей (на будущее)
        cursor.execute("PRAGMA foreign_keys = ON")

        # Создание таблицы vineyard_features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vineyard_features (
                osm_id INTEGER PRIMARY KEY,   -- Уникальный ID из OpenStreetMap
                lat REAL NOT NULL,            -- Широта
                lon REAL NOT NULL,            -- Долгота
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP          -- Время последнего обновления записи
            );
        ''')

        # Создание таблицы negative_features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS negative_features (
                osm_id INTEGER PRIMARY KEY,   -- Уникальный ID из OpenStreetMap
                lat REAL NOT NULL,            -- Широта
                lon REAL NOT NULL,            -- Долгота
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP          -- Время последнего обновления записи
            );
        ''')

        # сохранение изменений и закрытие БД
        conn.commit()
        conn.close()

        print("База данных и таблицы успешно созданы!\nТаблицы (vineyard_features, negative_features)")
    except sqlite3.Error as e:
        print(f"Ошибка при работе с SQLite: {e}")


if __name__ == "__main__":
    # Путь к БД
    db_folder_path = Path(__file__).resolve().parent.parent / "data"
    db_folder_path.mkdir(exist_ok=True)
    db_name = 'vineyard_1.db'
    db_path = db_folder_path / db_name
    print("Указанный путь до БД:\t", db_path)

    setup_database(db_path)


# elevation_GEE_USGS_30m REAL,                         -- Высота над уровнем моря
# elevation_GEE_USGS_30m_status TEXT DEFAULT 'pending',-- Статус обработки высоты
#
# slope_GEE_USGS_30m REAL,                             -- Уклон (крутизна)
# slope_GEE_USGS_30m_status TEXT DEFAULT 'pending',    -- Статус обработки уклона
#
# aspect_GEE_USGS_30m REAL,                            -- Экспозиция (куда смотрит склон)
# aspect_GEE_USGS_30m_status TEXT DEFAULT 'pending',   -- Статус обработки экспозиции
#
# hillshade_GEE_USGS_30m REAL,                         -- Теневой рельеф
# hillshade_GEE_USGS_30m_status TEXT DEFAULT 'pending' -- Статус теневой рельеф

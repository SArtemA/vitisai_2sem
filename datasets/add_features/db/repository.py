# repository.py - Загрузка в БД данных
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Union


def insert_points(points, db_path, table_name):
    """
    Функця для заполнения полей osm_id, lat, lon.

    Args:
        points: Список кортежей (osm_id, lat, lon)
        db_path: Путь к файлу базы данных
        table_name: Название таблицы ('vineyard_features' или 'negative_features')
    """

    allowed_tables = {'vineyard_features', 'negative_features'}
    if table_name not in allowed_tables:
        raise ValueError(f"Недопустимое имя таблицы. Разрешены: {allowed_tables}")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Динамическое выполнение вставки с указанием таблицы
        cursor.executemany(f"""
            INSERT OR IGNORE INTO {table_name} (osm_id, lat, lon)
            VALUES (?, ?, ?)
        """, points)

        conn.commit()
        print(f"Успешно вставлено {len(points)} записей в таблицу '{table_name}'")

    except sqlite3.Error as e:
        print(f"Ошибка при работе с SQLite: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def get_row_by_status(db_path, cols_filter, status='pending', AND_or_OR="AND", limit=1000):
    """
    Функция для получения osm_id, lat, lon если "cols_filter" в статусе {status}.

    Args:
        db_path: Путь к БД.
        cols_filter: Колонки которые надо првоерить, на вход либо str, либо list[str, str, ...].
        limit=1000: Лимит вывода значений.

    Returns:
        list[osm_id, lat, lon].
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if isinstance(cols_filter, str):
        cols_filter = [cols_filter]
    else:
        cols_filter = list(dict.fromkeys(cols_filter))

    for i in range(len(cols_filter)):
        if not cols_filter[i].endswith('_status'):
            cols_filter[i] = cols_filter[i] + '_status'

    conditions = f" {AND_or_OR} ".join([f"{col} = '{status}'" for col in cols_filter])

    query = f"""
        SELECT osm_id, lat, lon
        FROM vineyard_features
        WHERE {conditions}
        LIMIT ?
    """

    try:
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    return rows


def update_vineyard_features(
    db_path: str,
    id_in_db: Union[int, List[int]],
    features: Union[Dict, List[Dict]],
    status: str = 'done'
) -> bool:
    """
    Функция для обновления данных. Поддерживает одиночное и массовое обновление.

    Args:
        db_path: Путь к БД.
        id_in_db: Один ID или список ID.
        features: Один словарь признаков или список словарей.
        status: Какой статус будет выставляться.

    Returns:
        bool: Успешность операции.
    """
    # Приводим к спискам для единообразной обработки
    if not isinstance(id_in_db, list):
        ids = [id_in_db]
        features_list = [features]
    else:
        ids = id_in_db
        features_list = features if isinstance(features, list) else [features] * len(ids)

    # Проверяем соответствие длин
    if len(ids) != len(features_list):
        print(f"Ошибка: количество ID ({len(ids)}) не соответствует количеству наборов признаков ({len(features_list)})")
        return False

    if not ids or not features_list:
        return False

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Обновляем каждый объект
            for osm_id, feat in zip(ids, features_list):
                if not feat:  # Пропускаем пустые словари
                    continue

                set_clauses = []
                params = []

                for column, value in feat.items():
                    if value == 'error':
                        # При ошибке не меняем значение, только статус
                        set_clauses.append(f"{column}_status = ?")
                        params.append('error')
                    elif value == 'NULL' or value is None:
                        pass
                    else:
                        set_clauses.append(f"{column} = ?")
                        set_clauses.append(f"{column}_status = ?")
                        params.extend([value, status])

                if not set_clauses:  # Если нечего обновлять
                    continue

                set_clauses.append("updated_at = ?")
                params.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                params.append(osm_id)

                query = f"""
                    UPDATE vineyard_features SET {', '.join(set_clauses)}
                    WHERE osm_id = ?
                """

                cursor.execute(query, params)

            conn.commit()
            return True

    except sqlite3.Error as e:
        print(f"Ошибка при обновлении базы данных: {e}")
        return False
        # Здесь можно реализовать логику записи ошибки в лог или смены статуса на 'error'


def reset_row_dynamically(db_path, osm_id) -> bool:
    """
    Автоматически находит все столбцы в таблице и обнуляет их,
    учитывая их тип (NULL для данных, 'pending' для статусов).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Информация о столбцах таблицы
            table_name = "vineyard_features"
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()

            set_parts = []
            # Технические поля, которые НЕ трогаем
            exclude = ['osm_id', 'lat', 'lon', 'created_at']

            for col in columns:
                col_name = col['name']

                if col_name in exclude:
                    continue

                # 2. Логика сброса:
                if col_name == 'updated_at':
                    set_parts.append(f"{col_name} = CURRENT_TIMESTAMP")
                elif col_name.endswith('_status'):
                    set_parts.append(f"{col_name} = 'pending'")
                else:
                    set_parts.append(f"{col_name} = NULL")

            if not set_parts:
                return False

            # 3. Собираем и выполняем запрос
            sql = f"UPDATE vineyard_features SET {', '.join(set_parts)} WHERE osm_id = ?"
            cursor.execute(sql, (osm_id,))
            conn.commit()
            print(f"Запись {osm_id} динамически обновлена.")
        return True

    except sqlite3.Error as e:
        print(f"Ошибка: {e}")
        return False


def create_feature_cols(db_path, col_name):
    """
    Создание колонки для параметра и статуса 'panding'.
    Если колонка с признаком есть, а со статусом нету, то она её создаст

    Args:
        db_path: Путь к БД.
        col_name: str или list[str, str, ...].
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if isinstance(col_name, str):
        columns = [col_name]
    else:
        columns = col_name

    for col in columns:
        try:
            # 1. Создаем основную колонку для данных (например, REAL для координат/высот)
            # Если данные могут быть разными, можно использовать BLOB или оставить тип гибким
            cursor.execute(f"ALTER TABLE vineyard_features ADD COLUMN {col} REAL")
            print(f"Колонка '{col}' успешно добавлена.")
        except sqlite3.OperationalError:
            print(f"Колонка '{col}' уже существует.")

        try:
            # 2. Создаем колонку статуса со значением 'pending' по умолчанию
            status_col = f"{col}_status"
            cursor.execute(f"ALTER TABLE vineyard_features ADD COLUMN {status_col} TEXT DEFAULT 'pending'")
            print(f"Колонка '{status_col}' успешно добавлена.")
        except sqlite3.OperationalError:
            print(f"Колонка '{status_col}' уже существует.")

    conn.commit()
    conn.close()


def delete_feature_cols(db_path, col_name):
    """
    Удаляет колонку параметра и колонку его статуса.

    Args:
        db_path: Путь к БД.
        col_name: str или list[str] с названиями признаков.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Приводим к списку
    columns = [col_name] if isinstance(col_name, str) else col_name

    for col in columns:
        # Список колонок для удаления (сама переменная + её статус)
        cols_to_remove = [col, f"{col}_status"]

        for target in cols_to_remove:
            try:
                # В SQLite нельзя удалить несколько колонок одним запросом
                # и нельзя использовать параметры (?) для имен колонок
                cursor.execute(f'ALTER TABLE vineyard_features DROP COLUMN "{target}"')
                print(f"Колонка '{target}' успешно удалена.")
            except sqlite3.OperationalError as e:
                # Если колонки нет, SQLite выдаст ошибку — перехватываем её
                print(f"Ошибка при удалении '{target}': {e}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Путь к БД
    print(sqlite3.sqlite_version)
    db_folder_path = Path(__file__).resolve().parent.parent / "data"
    db_folder_path.mkdir(exist_ok=True)
    db_name = 'vineyard_1.db'
    db_path = db_folder_path / db_name

    # TEST get_row_by_status
    # ans = get_row_by_status(
    #     db_path,
    #     "elevation_GEE_USGS_30m",  # ["elevation_GEE_USGS_30m_status", "slope_GEE_USGS_30m_status"],
    #     limit=5
    # )
    # print(ans)
    # print()
    # for i in ans:
    #     print(i)

    # TEST create_feature_cols
    # create_feature_cols(db_path, "test")

    # TSET delete_feature_cols
    # delete_feature_cols(db_path, "test")

    # TEST update_vineyard_features
    # id = 4812832
    # # data = {'aspect_GEE_USGS_30m': 0, 'elevation_GEE_USGS_30m': 260, 'hillshade_GEE_USGS_30m': 180, 'slope_GEE_USGS_30m': 1}
    # data = {'aspect_GEE_USGS_30m': 'error', 'elevation_GEE_USGS_30m': 'error', 'hillshade_GEE_USGS_30m': 'error', 'slope_GEE_USGS_30m': 'error'}
    # update_vineyard_features(db_path, id, data)

    # TEST reset_row_dynamically
    # id = 4812832
    # reset_row_dynamically(db_path, id)

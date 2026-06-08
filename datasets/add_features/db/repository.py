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


def get_row_by_status(
        db_path,
        table_name: str,
        cols_filter,
        status='pending',
        AND_or_OR="AND",
        limit=1000,
        random_order=False
    ):
    """
    Функция для получения osm_id, lat, lon если "cols_filter" в статусе {status}.

    Args:
        db_path: Путь к БД.
        cols_filter: Колонки которые надо првоерить, на вход либо str, либо list[str, str, ...].
        limit=1000: Лимит вывода значений.
        random_order=False: Включение случайного порядка строк.

    Returns:
        list[osm_id, lat, lon]?.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if isinstance(cols_filter, str):
        cols_filter = [cols_filter]
    # else:
    #     cols_filter = list(dict.fromkeys(cols_filter))

    if isinstance(status, str):
        status = [status]
    # else:
    #     status = list(dict.fromkeys(status))

    if len(cols_filter) != len(status):
        raise ValueError(
            f"Количество фильтров ({len(cols_filter)}) не совпадает "
            f"с количеством статусов ({len(status)})."
        )

    for i in range(len(cols_filter)):
        if not cols_filter[i].endswith('_status'):
            cols_filter[i] = cols_filter[i] + '_status'

    conditions = f" {AND_or_OR} ".join([f"{col} = '{stat}'" for col, stat in zip(cols_filter, status)])

    # Динамически добавляем ORDER BY RANDOM() при True
    order_clause = "ORDER BY RANDOM()" if random_order else ""

    query = f"""
        SELECT osm_id, lat, lon
        FROM {table_name}
        WHERE {conditions}
        {order_clause}
        LIMIT ?
    """

    try:
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
    finally:
        conn.close()

    return rows


def update_vineyard_features(
    db_path,
    table_name: str,
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
                    if value == 'error' or value == 'None' or value == 'NULL':
                        # При ошибке не меняем значение, только статус
                        set_clauses.append(f"{column}_status = ?")
                        params.append('error')
                    # elif value == 'NULL' or value is None:
                    #     pass
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
                    UPDATE {table_name} SET {', '.join(set_clauses)}
                    WHERE osm_id = ?
                """

                cursor.execute(query, params)

            conn.commit()
            return True

    except sqlite3.Error as e:
        print(f"Ошибка при обновлении базы данных: {e}")
        return False
        # Здесь можно реализовать логику записи ошибки в лог или смены статуса на 'error'


def reset_row_dynamically(
        db_path,
        table_name:str,
        osm_id
    ) -> bool:
    """
    Автоматически находит все столбцы в таблице и обнуляет их,
    учитывая их тип (NULL для данных, 'pending' для статусов).
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Информация о столбцах таблицы
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


def create_feature_cols(
        db_path,
        table_name,
        col_name
    ):
    """
    Создание колонки для параметра и статуса 'pending'.
    !!! Колонки для статусов не подавать !!!.
    Если колонка с признаком есть, а со статусом нету, то она её создаст

    Args:
        db_path: Путь к БД.
        col_name: str или list[str, str, ...].
        table_name: Имя таблицы в БД.
    """
    # Используем контекстный менеджер, чтобы соединение всегда закрывалось корректно
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Приводим к единому типу list
        columns_to_add = [col_name] if isinstance(col_name, str) else col_name

        # 1. Получаем список ВСЕХ колонок, которые СЕЙЧАС реально есть в таблице
        try:
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            # row[1] — это гарантированно текстовое имя колонки в SQLite
            existing_columns = {row[1] for row in cursor.fetchall()}
        except sqlite3.OperationalError as e:
            print(f"Ошибка: Не удалось прочитать таблицу [{table_name}]: {e}")
            return

        # 2. Проверяем и добавляем только недостающие колонки
        for col in columns_to_add:
            status_col = f"{col}_status"

            # Проверка и добавление основной колонки
            if col not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} REAL")
                    print(f"{table_name}: Колонка '{col}' успешно добавлена.")
                except sqlite3.OperationalError as e:
                    print(f"{table_name}: Ошибка при добавлении '{col}': {e}")
            else:
                print(f"{table_name}: Колонка '{col}' уже существует в БД (пропущено).")

            # Проверка и добавление колонки статуса
            if status_col not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {status_col} TEXT DEFAULT 'pending'")
                    print(f"{table_name}: Колонка '{status_col}' успешно добавлена.")
                except sqlite3.OperationalError as e:
                    print(f"{table_name}: Ошибка при добавлении '{status_col}': {e}")
            else:
                print(f"{table_name}: Колонка '{status_col}' уже существует в БД (пропущено).")


def delete_feature_cols(
        db_path,
        table_name,
        col_name
    ):
    """
    Удаляет колонку параметра и колонку его статуса.

    Args:
        db_path: Путь к БД.
        table_name: Название таблицы.
        col_name: str или list[str] с названиями признаков.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Получаем список ВСЕХ колонок, которые СЕЙЧАС реально есть в таблице
    try:
        cursor.execute(f"PRAGMA table_info([{table_name}])")
        # row[1] — это гарантированно текстовое имя колонки в SQLite
        existing_columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.OperationalError as e:
        print(f"Ошибка: Не удалось прочитать таблицу [{table_name}]: {e}")
        return

    # Приводим к списку
    columns = [col_name] if isinstance(col_name, str) else col_name

    for col in columns:
        # Список колонок для удаления (сама переменная + её статус)
        cols_to_remove = [col, f"{col}_status"]

        for target in cols_to_remove:
            if target in existing_columns:
                try:
                    # В SQLite нельзя удалить несколько колонок одним запросом
                    # и нельзя использовать параметры (?) для имен колонок
                    cursor.execute(f"ALTER TABLE [{table_name}] DROP COLUMN [{target}]")
                    print(f"Колонка '{target}' успешно удалена.")
                except sqlite3.OperationalError as e:
                    # Если колонки нет, SQLite выдаст ошибку — перехватываем её
                    print(f"Ошибка при удалении '{target}': {e}")
            else:
                print(f"В таблице '{table_name}' нету колонки '{target}'.")

    conn.commit()
    conn.close()


def get_count_row(
    db_path,
    table_name: str
    ) -> int:
    """Возвращает общее кол-во строк."""
    query = f"SELECT COUNT(*) FROM {table_name};"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        return result[0] if result else 0


def get_count_row_by_status(
        db_path,
        table_name: str,
        cols_filter,
        status='pending',
        AND_or_OR="AND"
    ):
    """
    Возвращает количество строк в таблице SQLite, удовлетворяющих условиям
    статуса для заданных колонок.

    Args:
        db_path: Путь к БД.
        table_name: Название таблицы.
        cols_filter: Колонки которые надо проверить,
            на вход либо str, либо list[str, str, ...].
        status: Значение статуса для фильтрации (по умолчанию 'pending').
        AND_or_OR: Логический оператор для объединения условий.
            Допустимы только 'AND' или 'OR'.

    Returns:
        int: Количество найденных строк.
    """
    # Валидация логического оператора во избежание синтаксических ошибок SQL
    AND_or_OR = AND_or_OR.upper().strip()
    if AND_or_OR not in ("AND", "OR"):
        raise ValueError("Параметр AND_or_OR должен быть равен 'AND' или 'OR'")

    # Нормализация входных колонок и удаление дубликатов
    if isinstance(cols_filter, str):
        columns = [cols_filter]
    else:
        columns = list(dict.fromkeys(cols_filter))

    if not columns:
        return 0

    # Формирование корректных имён колонок с суффиксом '_status'
    processed_cols = []
    for col in columns:
        if not col.endswith("_status"):
            col = col + "_status"
        # Экранируем имя колонки двойными кавычками для безопасности SQLite
        processed_cols.append(f'"{col.replace('"', '""')}"')

    # Экранируем имя таблицы
    safe_table_name = f'"{table_name.replace('"', '""')}"'

    # Построение безопасного SQL-запроса с использованием плейсхолдеров '?'
    conditions = f" {AND_or_OR} ".join([f"{col} = ?" for col in processed_cols])
    query = f"""
            SELECT COUNT(*)
            FROM {safe_table_name}
            WHERE {conditions}
        """

    # Подготавливаем список аргументов (статус дублируется под каждую колонку)
    params = [status] * len(processed_cols)

    # Выполнение запроса в безопасном контексте соединения
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()

        # Возвращаем само число (первый элемент кортежа), либо 0 если результат пустой
        return result[0] if result else 0


if __name__ == "__main__":
    # Путь к БД
    print(sqlite3.sqlite_version)
    db_folder_path = Path(__file__).resolve().parent.parent / "data"
    db_folder_path.mkdir(exist_ok=True)
    db_name = 'vineyard_1.db'
    db_path = db_folder_path / db_name
    print(db_path)

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
    delete_feature_cols(db_path,
        'negative_features',
        ['NDVI_phase1_2024',
        'NDVI_phase2_2024',
        'NDVI_phase3_2024',
        'NDWI_phase2_2024',
        'SAVI_phase2_2024',
        'SR_B2_mean_2024',
        'SR_B3_mean_2024',
        'SR_B4_mean_2024',
        'SR_B5_mean_2024',
        'SR_B6_mean_2024',
        'SR_B7_mean_2024',
        'ST_B10_mean_2024',
        'cloud_cover_phase2_2024']
        )

    # TEST update_vineyard_features
    # id = 4812832
    # # data = {'aspect_GEE_USGS_30m': 0, 'elevation_GEE_USGS_30m': 260, 'hillshade_GEE_USGS_30m': 180, 'slope_GEE_USGS_30m': 1}
    # data = {'aspect_GEE_USGS_30m': 'error', 'elevation_GEE_USGS_30m': 'error', 'hillshade_GEE_USGS_30m': 'error', 'slope_GEE_USGS_30m': 'error'}
    # update_vineyard_features(db_path, id, data)

    # TEST reset_row_dynamically
    # id = 4812832
    # reset_row_dynamically(db_path, id)

import sqlite3
from pathlib import Path
import shutil


def create_copy_db_by_status(src_db_path, dst_db_path, status_value='done', AND_or_OR='AND'):
    """
    Создаёт копию БД, перенося только строки, где все '*_status' столбецы
    содержат указанное значение (по умолчанию 'done').
    """
    # 1. Копируем структуру таблиц (без данных)
    shutil.copy2(src_db_path, dst_db_path)

    # 2. Подключаемся к копии БД для удаления данных и вставки отфильтрованных строк
    src_conn = sqlite3.connect(src_db_path)
    src_conn.row_factory = sqlite3.Row  # чтобы удобно обращаться к столбцам по именам
    dst_conn = sqlite3.connect(dst_db_path)

    tables = get_tables_name(src_db_path)

    for table in tables:
        # 2.1. Получаем список всех столбцов таблицы
        columns = get_columns_name(src_db_path, table)
        # Выделяем столбцы со статусом (оканчиваются на '_status')
        status_columns = [col for col in columns if col.endswith('_status')]

        if not status_columns:
            # Если в таблице нет статусных столбцов — пропускаем её (или можно копировать все строки)
            # По вашему описанию, я предполагаю, что нужно оставить таблицу пустой в новой БД,
            # но если хотите копировать все строки — раскомментируйте строки ниже.
            # В текущей версии — оставляем пустой, т.к. нет условий фильтра.
            # Если же нужно копировать все строки, используйте: dst_conn.execute(f"DELETE FROM {table}")
            # и затем вставку всех данных из src.
            dst_conn.execute(f"DELETE FROM {table}")
            continue

        # 2.2. Удаляем все строки из таблицы в копии (т.к. shutil.copy перенёс и данные)
        dst_conn.execute(f"DELETE FROM {table}")

        # 2.3. Формируем WHERE условие
        where_clause = f" {AND_or_OR} ".join([f"{col} = ?" for col in status_columns])

        # 2.4. Выбираем строки из исходной БД, подходящие под условие
        query = f"SELECT * FROM {table} WHERE {where_clause}"
        rows = src_conn.execute(query, [status_value] * len(status_columns)).fetchall()

        if not rows:
            continue

        # 2.5. Вставляем отфильтрованные строки в таблицу-копию
        placeholders = ", ".join(["?"] * len(columns))
        insert_query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"

        for row in rows:
            dst_conn.execute(insert_query, [row[col] for col in columns])

        dst_conn.commit()
        print(f"Таблица '{table}': перенесено {len(rows)} строк (со статусом '{status_value}')")

    src_conn.close()
    dst_conn.close()
    print(f"Готово! Новая БД: {dst_db_path}")


def get_tables_name(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")

    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not isinstance(tables, list):
        tables = [tables]

    return tables

def get_columns_name(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"""
        SELECT * 
        FROM {table_name}
        LIMIT 1
    """

    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    conn.close()

    return columns

if __name__ == "__main__":
    # Путь к папке
    db_folder_path = Path(__file__).resolve().parent.parent / "data"
    if db_folder_path.exists():
        print(f"Folder path: {db_folder_path}")
    else:
        raise FileNotFoundError(f"Folder path: {db_folder_path} - папка НЕ существует")

    # Основная БД
    db_main_name = 'vineyard_1.db'
    db_main_path = db_folder_path / db_main_name
    if not db_main_path.exists():
        raise FileNotFoundError(f"Основная БД: {db_folder_path} - НЕ существует")

    # Куда переносим
    db_second_name = 'vineyard_1_stat_done.db'
    db_second_path = db_folder_path / db_second_name

    all_tables = get_tables_name(db_main_path)
    print(f"Все таблицы:\t{all_tables}")

    # columns = get_columns_name(db_main_path, all_tables[0])
    # status_columns = [item for item in columns if item.endswith('_status')]

    create_copy_db_by_status(db_main_path, db_second_path)
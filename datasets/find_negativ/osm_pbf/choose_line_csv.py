from pathlib import Path
import pandas as pd


def sample_csv(
    input_file: Path,
    output_file: Path,
    n_rows: int = None,
    frac: float = None,
    random_state: int = 42,
):
    """Функция для случайной выборки строк из CSV-файла с использованием pathlib.

    :param input_file: Путь к исходному CSV-файлу (объект Path).
    :param output_file: Путь для сохранения результата (объект Path).
    :param n_rows: Точное количество строк, которые нужно выбрать.
    :param frac: Доля от общего объема строк (от 0.0 до 1.0).
    :param random_state: Сид для воспроизводимости.
    """
    # Проверяем, существует ли исходный файл
    if not input_file.exists():
        raise FileNotFoundError(f"Исходный файл не найден по пути: {input_file}")

    # 1. Читаем исходный файл (pandas отлично принимает объекты Path)
    df = pd.read_csv(input_file)

    # 2. Делаем выборку
    if n_rows is not None:
        sampled_df = df.sample(
            n=min(n_rows, len(df)), random_state=random_state
        )
        print(f"Успешно выбрано {len(sampled_df)} строк (запрошено {n_rows}).")

    elif frac is not None:
        sampled_df = df.sample(frac=frac, random_state=random_state)
        print(
            f"Успешно выбрано {len(sampled_df)} строк (применили долю {frac*100}%)."
        )

    else:
        raise ValueError("Нужно указать либо параметр n_rows, либо frac!")

    # Создаем родительские папки для выходного файла, если они вдруг не существуют
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 3. Сохраняем результат
    sampled_df.to_csv(output_file, index=False)
    print(f"Результат успешно сохранен в: {output_file.resolve()}")


# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===
if __name__ == "__main__":
    # Определяем базовую папку (например, текущую директорию скрипта)
    BASE_DIR = Path(__file__).resolve().parent

    # Задаем пути через Path (можно собирать пути через слэш `/`)
    input_csv = BASE_DIR / "output_polygons.csv"
    output_csv = BASE_DIR / "choose_farmland.csv"

    try:
        sample_csv(
            input_file=input_csv,
            output_file=output_csv,
            n_rows=1500,
            random_state=2026,
        )
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
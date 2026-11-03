# VitiPredict ML Extended

**VitiPredict ML Extended** — расширенная версия веб-приложения для анализа пригодности территории под выращивание винограда и подбора подходящих сортов винограда на основе координат, геопространственных данных и моделей машинного обучения.

Ветка `ML_extended` развивает базовую версию проекта: кроме бинарного прогноза пригодности участка, приложение также формирует рекомендации по сортам винограда и использует расширенный набор экологических признаков.


---

<video src="https://github.com/user-attachments/assets/19bfd6a6-4d5a-494a-beaf-c39e77eee0c5" width="100%" controls></video>

---

## Содержание

- [Описание проекта](#описание-проекта)
- [Отличия ветки ML_extended](#отличия-ветки-ml_extended)
- [Основные возможности](#основные-возможности)
- [Как работает система](#как-работает-система)
- [Технологический стек](#технологический-стек)
- [Структура проекта](#структура-проекта)
- [Описание основных файлов](#описание-основных-файлов)
- [Установка и запуск](#установка-и-запуск)
- [Настройка Google Earth Engine](#настройка-google-earth-engine)
- [API](#api)
- [База данных](#база-данных)
- [ML-модели](#ml-модели)
- [Экологические признаки](#экологические-признаки)
- [Работа с моделями](#работа-с-моделями)
- [Диагностика базы данных](#диагностика-базы-данных)
- [Настройки проекта](#настройки-проекта)
- [Возможные проблемы](#возможные-проблемы)
- [Автор](#автор)
- [Лицензия](#лицензия)

---

## Описание проекта

**VitiPredict ML Extended** — учебный проект, предназначенный для интеллектуальной оценки земельного участка под виноградарство.

Пользователь выбирает точку на карте или вводит координаты вручную. После этого приложение:

1. получает экологические и геопространственные признаки выбранной точки;
2. определяет, подходит ли участок для виноградарства;
3. при положительном результате формирует рекомендации по сортам винограда;
4. сохраняет координаты, признаки и результат в локальную SQLite-базу данных.

Проект объединяет веб-интерфейс, FastAPI backend, SQLite-базу, Google Earth Engine и модели машинного обучения на XGBoost.

---

## Отличия ветки `ML_extended`

Ветка `ML_extended` отличается от базовой версии проекта следующими изменениями:

- структура проекта разделена на модули `databases/` и `models/`;
- добавлен файл `requirements.txt`;
- база данных находится в папке `databases/`;
- используется база `vineyards_v3.db`;
- расширен набор признаков для анализа участка;
- добавлены признаки почвы, климата, растительности, ветра, солнечной радиации и fire risk;
- добавлена бинарная модель пригодности участка;
- добавлена multi-label модель для рекомендаций сортов винограда;
- endpoint `/api/predict` возвращает не только `suitable`, но и `recommendations`;
- локальный запуск выполняется на порту `8000`.

---

## Основные возможности

- отображение карты с точками виноградников;
- выбор координат на карте;
- ручной ввод широты и долготы;
- сбор расширенных экологических признаков;
- прогноз пригодности участка под виноградник;
- рекомендации по сортам винограда;
- сохранение результатов в SQLite;
- автоматическая миграция схемы базы при добавлении новых колонок;
- диагностика базы данных через отдельный скрипт;
- fallback-режим при недоступности Google Earth Engine.

---

## Как работает система

Общий сценарий работы:

```text
Пользователь
   │
   ▼
HTML / Jinja2 интерфейс
   │
   ▼
FastAPI backend
   │
   ├── получает координаты
   ├── вызывает data_fetcher.py
   ├── получает экологические признаки
   ├── вызывает бинарную ML-модель
   ├── если участок подходит — вызывает модель рекомендаций
   ├── сохраняет результат в SQLite
   └── возвращает JSON-ответ frontend-части
```

Подробный сценарий:

1. Пользователь открывает приложение.
2. На странице карты выбирает точку или вводит координаты вручную.
3. Frontend отправляет POST-запрос на `/api/predict`.
4. Backend получает координаты через Pydantic-модель `CoordinatesIn`.
5. `data_fetcher.py` собирает признаки через Google Earth Engine.
6. Если Google Earth Engine недоступен, возвращается fallback-набор нулевых значений.
7. `BinSuitClassifier` определяет пригодность участка.
8. Если участок подходит, `MultiGrapeXGBClassifier` формирует рейтинг сортов винограда.
9. Результат сохраняется в таблицу `vineyard_features`.
10. Frontend показывает пригодность, рекомендации и параметры среды.

---

## Технологический стек

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- SQLAlchemy

### Frontend

- HTML
- Jinja2 Templates
- JavaScript

### Data / Geo

- Google Earth Engine
- SRTM
- ERA5-Land
- OpenLandMap
- ESA WorldCover
- MODIS
- TerraClimate

### Machine Learning

- XGBoost
- scikit-learn
- MultiOutputClassifier
- StandardScaler
- pandas
- NumPy
- joblib

### Database

- SQLite
- SQLAlchemy ORM

---

## Структура проекта

```text
.
├── databases/
│   ├── __init__.py
│   ├── database.py
│   ├── db_inspector.py
│   └── vineyards_v3.db
│
├── models/
│   └── ml_model.py
│
├── templates/
│   ├── base.html
│   ├── map.html
│   └── predict.html
│
├── data_fetcher.py
├── main.py
├── requirements.txt
├── run.py
└── .gitignore
```

> Примечание: обученные модели должны находиться в подпапках внутри `models/`, если они были сгенерированы локально.

---

## Описание основных файлов

### `main.py`

Главный файл FastAPI-приложения.

Содержит:

- инициализацию FastAPI;
- подключение шаблонов Jinja2;
- загрузку ML-моделей при старте приложения;
- маршруты frontend-страниц;
- API для получения виноградников;
- API для прогноза пригодности участка;
- вызов рекомендаций сортов;
- сохранение результатов анализа в базу данных.

Основные маршруты:

```text
GET  /
GET  /predict_page
GET  /api/vineyards
POST /api/predict
```

---

### `run.py`

Файл локального запуска приложения.

Он:

- импортирует объект `app` из `main.py`;
- запускает Uvicorn;
- открывает приложение в браузере;
- использует хост `127.0.0.1`;
- использует порт `8000`.

После запуска приложение доступно по адресу:

```text
http://127.0.0.1:8000
```

---

### `data_fetcher.py`

Модуль получения экологических данных по координатам.

Он использует Google Earth Engine и собирает признаки из нескольких источников:

- SRTM — рельеф;
- ERA5-Land — климат и погодные параметры;
- OpenLandMap — почвенные признаки;
- ESA WorldCover — тип земного покрова;
- MODIS — индексы растительности и влажности;
- TerraClimate — показатель засушливости / fire risk proxy.

Если Google Earth Engine недоступен, используется fallback-функция `_fetch_from_public_apis`, которая возвращает нулевые значения и статусы `FAILED`.

---

### `databases/database.py`

Модуль базы данных.

Содержит:

- путь к SQLite-базе;
- SQLAlchemy engine;
- session factory;
- ORM-модель `VineyardFeature`;
- автоматическое создание таблицы;
- автоматическую миграцию схемы при появлении новых колонок.

Используемая база данных:

```text
databases/vineyards_v3.db
```

Основная таблица:

```text
vineyard_features
```

---

### `databases/db_inspector.py`

Скрипт для диагностики базы данных.

Позволяет:

- вывести структуру таблиц;
- посмотреть колонки;
- вывести несколько первых записей;
- подсчитать количество подходящих и неподходящих участков;
- получить базовую статистику по базе.

Запуск:

```bash
python databases/db_inspector.py
```

---

### `models/ml_model.py`

Основной модуль машинного обучения.

Содержит два класса:

```python
BinSuitClassifier
```

Бинарная модель, которая определяет пригодность участка:

```text
0 — участок не подходит
1 — участок подходит
```

```python
MultiGrapeXGBClassifier
```

Multi-label модель, которая формирует рейтинг подходящих сортов винограда.

---

### `templates/base.html`

Базовый HTML-шаблон приложения.

Используется как общий layout для страниц.

---

### `templates/map.html`

Страница карты.

Отображает карту предсказания зон для виноделия и загружает точки виноградников.

---

### `templates/predict.html`

Страница анализа координат.

Содержит:

- выбор локации;
- ввод широты и долготы;
- кнопку анализа;
- блок рекомендаций по сортам винограда;
- блок условий окружающей среды.

---

### `requirements.txt`

Файл зависимостей проекта.

Позволяет установить все основные библиотеки одной командой:

```bash
pip install -r requirements.txt
```

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SArtemA/vitisai_2sem.git
cd vitisai_2sem
```

### 2. Переключение на ветку `ML_extended`

```bash
git checkout ML_extended
```

Если ветка ещё не загружена локально:

```bash
git fetch origin
git checkout ML_extended
```

### 3. Создание виртуального окружения

```bash
python -m venv venv
```

### 4. Активация виртуального окружения

Для Windows:

```bash
venv\Scripts\activate
```

Для Linux / macOS:

```bash
source venv/bin/activate
```

### 5. Установка зависимостей

```bash
pip install -r requirements.txt
```

Если установка через `requirements.txt` не сработала, можно установить основные зависимости вручную:

```bash
pip install fastapi uvicorn sqlalchemy pydantic pandas numpy scikit-learn xgboost joblib earthengine-api requests
```

### 6. Запуск приложения

```bash
python run.py
```

После запуска откройте:

```text
http://127.0.0.1:8000
```

При запуске через `run.py` браузер должен открыться автоматически.

---

## Настройка Google Earth Engine

Проект использует Google Earth Engine для получения геопространственных и экологических данных.

Установите Earth Engine API:

```bash
pip install earthengine-api
```

Выполните авторизацию:

```bash
earthengine authenticate
```

В `data_fetcher.py` используется проект:

```python
ee.Initialize(project='pp-2-sem-grapes')
```

Если используется другой Google Cloud / Earth Engine project, замените значение `project`:

```python
ee.Initialize(project='your-project-id')
```

Если Google Earth Engine недоступен, приложение продолжит работать, но экологические признаки будут заполнены нулевыми значениями. В этом случае качество прогноза и рекомендаций будет некорректным.

---

## API

### Получение списка сохранённых точек

```http
GET /api/vineyards
```

Endpoint возвращает записи из таблицы `vineyard_features`.

Пример ответа:

```json
[
  {
    "osm_id": 1,
    "lat": 44.6167,
    "lon": 33.5254,
    "elevation": 120.0,
    "slope": 8.2,
    "aspect": 180.0,
    "hillshade": 210.0,
    "mid_year_temp": 25.1,
    "precipitation": 420.0,
    "ndvi": 0.61,
    "ndwi": 0.18,
    "solar_radiation": 15.4,
    "humidity": 12.1,
    "wind_speed": 3.2,
    "soil_ph": 6.7,
    "soil_organic_carbon": 18.0,
    "fire_risk": -0.5,
    "is_suitable": true
  }
]
```

---

### Прогноз пригодности участка и рекомендации сортов

```http
POST /api/predict
```

Тело запроса:

```json
{
  "lat": 44.6167,
  "lon": 33.5254
}
```

Пример ответа:

```json
{
  "suitable": true,
  "data_collected": {
    "elevation": 120.0,
    "slope": 8.2,
    "aspect": 180.0,
    "hillshade": 210.0,
    "mid_year_temp": 25.1,
    "precipitation": 420.0,
    "humidity": 12.1,
    "solar_radiation": 15.4,
    "wind_speed": 3.2,
    "evapotranspiration": 0.002,
    "evi": 0.42,
    "lai": 2.1,
    "land_cover_type": 40,
    "soil_ph": 6.7,
    "soil_organic_carbon": 18.0,
    "fire_risk": -0.5,
    "winkler_index": 1650.0,
    "ndvi": 0.61,
    "ndwi": 0.18,
    "elevation_status": "SUCCESS",
    "slope_status": "SUCCESS",
    "aspect_status": "SUCCESS",
    "hillshade_status": "SUCCESS"
  },
  "recommendations": [
    {
      "grape": "arinto",
      "score": 87.35
    },
    {
      "grape": "mostosa",
      "score": 76.12
    }
  ]
}
```

Если участок признан неподходящим, список `recommendations` может быть пустым.

---

## База данных

Проект использует SQLite.

Путь к базе данных:

```text
databases/vineyards_v3.db
```

Основная таблица:

```text
vineyard_features
```

### Поля таблицы

| Поле | Тип | Описание |
|---|---:|---|
| `osm_id` | Integer | Первичный ключ |
| `lat` | Float | Широта |
| `lon` | Float | Долгота |
| `created_at` | DateTime | Дата создания записи |
| `updated_at` | DateTime | Дата обновления записи |
| `elevation` | Float | Высота над уровнем моря |
| `elevation_GEE_USGS_30m_status` | String | Статус получения высоты |
| `slope` | Float | Уклон поверхности |
| `slope_GEE_USGS_30m_status` | String | Статус получения уклона |
| `aspect` | Float | Экспозиция склона |
| `aspect_GEE_USGS_30m_status` | String | Статус получения экспозиции |
| `hillshade` | Float | Освещённость рельефа |
| `hillshade_GEE_USGS_30m_status` | String | Статус получения hillshade |
| `mid_year_temp` | Float | Средняя температура |
| `precipitation` | Float | Количество осадков |
| `ndvi` | Float | Индекс растительности |
| `ndwi` | Float | Индекс влажности |
| `solar_radiation` | Float | Солнечная радиация |
| `humidity` | Float | Влажность / proxy-показатель |
| `wind_speed` | Float | Скорость ветра |
| `evapotranspiration` | Float | Эвапотранспирация |
| `evi` | Float | Enhanced Vegetation Index |
| `lai` | Float | Leaf Area Index |
| `land_cover_type` | Integer | Тип земного покрова |
| `soil_ph` | Float | pH почвы |
| `soil_organic_carbon` | Float | Органический углерод в почве |
| `fire_risk` | Float | Proxy-показатель засушливости / fire risk |
| `is_suitable` | Boolean | Признак пригодности участка |

> В коде сбора данных также рассчитывается `winkler_index`. Если этот признак используется в ML-модели, стоит добавить соответствующую колонку в ORM-модель и базу данных.

---

## ML-модели

В ветке `ML_extended` используются две ML-модели.

---

### 1. Бинарная модель пригодности участка

Класс:

```python
BinSuitClassifier
```

Назначение:

```text
Определить, подходит ли участок для виноградарства.
```

Целевая переменная:

```text
is_suitable
```

Модель:

```text
XGBClassifier
```

Ожидаемые файлы модели:

```text
models/trained_models_bin/xgboost_bin_model.json
models/trained_models_bin/xgboost_bin_model_scaler.pkl
```

---

### 2. Multi-label модель рекомендаций сортов

Класс:

```python
MultiGrapeXGBClassifier
```

Назначение:

```text
Сформировать рейтинг сортов винограда по вероятности пригодности.
```

Модель:

```text
MultiOutputClassifier + XGBClassifier
```

Целевые сорта:

```text
arnsburger
arinto
mostosa
abbuoto
abouriou
acitana
```

Ожидаемые файлы модели:

```text
models/trained_models_multi/multi_output_xgb_model.joblib
models/trained_models_multi/scaler.joblib
models/trained_models_multi/feature_names.joblib
models/trained_models_multi/target_names.joblib
```

Результат работы модели:

```json
[
  {
    "grape": "arinto",
    "score": 87.35
  },
  {
    "grape": "mostosa",
    "score": 76.12
  }
]
```

---

## Экологические признаки

Ветка `ML_extended` использует расширенный набор признаков:

```text
elevation
slope
aspect
hillshade
mid_year_temp
precipitation
ndvi
ndwi
solar_radiation
humidity
wind_speed
evapotranspiration
evi
lai
land_cover_type
soil_ph
soil_organic_carbon
fire_risk
winkler_index
```

### Описание признаков

| Признак | Описание |
|---|---|
| `elevation` | Высота над уровнем моря |
| `slope` | Уклон поверхности |
| `aspect` | Направление склона |
| `hillshade` | Освещённость рельефа |
| `mid_year_temp` | Средняя температура |
| `precipitation` | Осадки |
| `ndvi` | Индекс растительности |
| `ndwi` | Индекс влажности |
| `solar_radiation` | Солнечная радиация |
| `humidity` | Влажность / температурный proxy |
| `wind_speed` | Скорость ветра |
| `evapotranspiration` | Эвапотранспирация |
| `evi` | Улучшенный индекс растительности |
| `lai` | Индекс листовой поверхности |
| `land_cover_type` | Тип земного покрова |
| `soil_ph` | Кислотность почвы |
| `soil_organic_carbon` | Органический углерод в почве |
| `fire_risk` | Proxy риска засухи / пожара |
| `winkler_index` | Сумма активных температур для виноградарства |

---

## Работа с моделями

### Загрузка моделей

При старте приложения в `main.py` создаются экземпляры:

```python
suit_class = BinSuitClassifier()
variety_model = MultiGrapeXGBClassifier()
```

Если файлы моделей отсутствуют, предсказания или рекомендации могут быть недоступны.

---

### Обучение бинарной модели

Пример ручного обучения из Python:

```python
import pandas as pd
from models.ml_model import BinSuitClassifier

df = pd.read_csv("your_dataset.csv")

model = BinSuitClassifier()
model.train(df, use_grid_search=False, device="cpu")
```

---

### Обучение multi-label модели сортов

```python
import pandas as pd
from models.ml_model import MultiGrapeXGBClassifier

df = pd.read_csv("your_dataset.csv")

model = MultiGrapeXGBClassifier()
model.train(df, use_grid_search=False, device="cpu")
```

---

### Использование GPU

В методах `train` есть параметр:

```python
device="cuda"
```

Пример:

```python
model.train(df, use_grid_search=False, device="cuda")
```

Для этого требуется корректно установленная версия XGBoost с поддержкой CUDA и подходящие драйверы.

---

## Диагностика базы данных

Для проверки базы используется:

```bash
python databases/db_inspector.py
```

Скрипт выводит:

- список таблиц;
- список колонок;
- количество записей;
- количество подходящих и неподходящих участков;
- базовую статистику.

---

## Настройки проекта

### Хост и порт

Файл:

```text
run.py
```

```python
_HOST = '127.0.0.1'
_PORT = 8000
```

---

### База данных

Файл:

```text
databases/database.py
```

```python
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'vineyards_v3.db')}"
```

---

### Google Earth Engine project

Файл:

```text
data_fetcher.py
```

```python
ee.Initialize(project='pp-2-sem-grapes')
```

---

### Пути к моделям

Файл:

```text
models/ml_model.py
```

Бинарная модель:

```text
models/trained_models_bin/
```

Multi-label модель:

```text
models/trained_models_multi/
```

---

## Возможные проблемы

### Google Earth Engine не активен

В консоли может появиться сообщение:

```text
GEE Not Active
```

Возможные причины:

- не выполнена авторизация;
- нет доступа к проекту Google Earth Engine;
- указан неверный project ID;
- отсутствует интернет-соединение.

Решение:

```bash
earthengine authenticate
```

Проверьте значение `project` в `data_fetcher.py`.

---

### Возвращаются только нулевые признаки

Это означает, что приложение перешло в fallback-режим.

Возможные причины:

- Google Earth Engine недоступен;
- ошибка авторизации;
- ошибка доступа к dataset;
- точка вне зоны покрытия отдельных источников;
- ошибка сетевого соединения.

В таком режиме прогнозы могут быть некорректными.

---

### Модель не найдена

Ожидаемые файлы бинарной модели:

```text
models/trained_models_bin/xgboost_bin_model.json
models/trained_models_bin/xgboost_bin_model_scaler.pkl
```

Ожидаемые файлы модели рекомендаций:

```text
models/trained_models_multi/multi_output_xgb_model.joblib
models/trained_models_multi/scaler.joblib
models/trained_models_multi/feature_names.joblib
models/trained_models_multi/target_names.joblib
```

Если этих файлов нет, необходимо обучить модели или добавить сохранённые артефакты модели в проект.

---

### Ошибка при установке зависимостей

Попробуйте обновить `pip`:

```bash
python -m pip install --upgrade pip
```

Затем повторите установку:

```bash
pip install -r requirements.txt
```

---

### Ошибка XGBoost

Проверьте установку:

```bash
pip install xgboost
```

Если используется GPU, проверьте CUDA, драйверы и совместимость XGBoost.

---

### Ошибка из-за `winkler_index`

В `models/ml_model.py` признак `winkler_index` входит в список `FEATURES`, а `data_fetcher.py` его возвращает. При этом в ORM-модели базы может отсутствовать колонка `winkler_index`.

Решение:

1. добавить колонку `winkler_index` в модель `VineyardFeature`;
2. выполнить автоматическую миграцию;
3. убедиться, что значение сохраняется в базе при создании новой записи.

Пример поля:

```python
winkler_index = Column(Float)
```

---

### Прогноз всегда одинаковый

Возможные причины:

- признаки возвращаются нулями из fallback-режима;
- модель обучена на малом наборе данных;
- в обучающей выборке несбалансированные классы;
- scaler обучен на другой структуре признаков;
- порядок признаков при обучении и прогнозе отличается.

---


## Пример полного сценария запуска

```bash
git clone https://github.com/SArtemA/vitisai_2sem.git
cd vitisai_2sem

git checkout ML_extended

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

earthengine authenticate

python run.py
```

После запуска:

```text
http://127.0.0.1:8000
```

---

## Автор


**Так же не забыть бы доьавить имена и ссылки**

GitHub: [https://github.com/SArtemA](https://github.com/SArtemA)

---

## Лицензия

А НУЖНА ЛИ???????
Лицензия в репозитории не указана.

Перед использованием проекта в коммерческих или публичных целях рекомендуется добавить файл `LICENSE`.

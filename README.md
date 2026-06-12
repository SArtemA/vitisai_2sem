# VitiPredict / Viticulture Predictor

**VitiPredict** — веб-приложение для анализа пригодности территории под выращивание винограда на основе координат, геопространственных данных и модели машинного обучения.

Проект позволяет выбрать точку на карте, получить экологические характеристики участка, выполнить ML-прогноз пригодности территории и сохранить результат анализа в локальную SQLite-базу данных.

---

## Содержание

- [Описание проекта](#описание-проекта)
- [Основные возможности](#основные-возможности)
- [Как работает система](#как-работает-система)
- [Технологический стек](#технологический-стек)
- [Структура проекта](#структура-проекта)
- [Описание основных файлов](#описание-основных-файлов)
- [Установка и запуск](#установка-и-запуск)
- [Настройка Google Earth Engine](#настройка-google-earth-engine)
- [API](#api)
- [База данных](#база-данных)
- [ML-модель](#ml-модель)
- [Миграция и обогащение данных](#миграция-и-обогащение-данных)
- [Настройки проекта](#настройки-проекта)
- [Автор](#автор)
- [Лицензия](#лицензия)

---

## Описание проекта

**VitiPredict** — учебный проект, направленный на оценку пригодности географической точки для виноградарства.

Приложение объединяет несколько частей:

1. веб-интерфейс для работы с картой;
2. backend на FastAPI;
3. локальную SQLite-базу данных;
4. модуль сбора экологических и геопространственных данных;
5. модель машинного обучения на XGBoost.

Пользователь вводит или выбирает координаты участка, после чего система получает набор признаков:

- высоту над уровнем моря;
- уклон поверхности;
- экспозицию склона;
- освещённость рельефа;
- температуру;
- количество осадков;
- NDVI;
- NDWI.

На основе этих данных ML-модель определяет, подходит ли выбранный участок для выращивания винограда.

---

## Основные возможности

- отображение карты с данными о виноградниках;
- выбор точки для анализа;
- ввод координат вручную;
- получение экологических и геопространственных характеристик по координатам;
- прогноз пригодности участка под виноградник;
- сохранение результатов анализа в SQLite-базу;
- автоматическая загрузка или обучение ML-модели;
- fallback-режим при недоступности Google Earth Engine;
- возможность переобучения модели на обновлённой базе данных.

---

## Как работает система

Общий сценарий работы приложения:

```text
Пользователь
   │
   ▼
Веб-интерфейс
   │
   ▼
FastAPI backend
   │
   ├── получает координаты
   ├── запрашивает экологические данные
   ├── передаёт признаки в ML-модель
   ├── получает прогноз
   └── сохраняет результат в SQLite
```

Более подробно:

1. Пользователь открывает страницу приложения.
2. Выбирает точку на карте или вводит координаты вручную.
3. Frontend отправляет координаты на endpoint `/api/predict`.
4. Backend вызывает модуль `data_fetcher.py`.
5. Модуль пытается получить данные через Google Earth Engine.
6. Если Google Earth Engine недоступен, используется fallback-режим.
7. Полученные признаки передаются в `ml_model.py`.
8. XGBoost-модель возвращает бинарный прогноз.
9. Результат сохраняется в таблицу `vineyard_features`.
10. Пользователь получает ответ о пригодности участка.

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
- SRTM / terrain features
- TerraClimate
- Landsat 8
- NDVI
- NDWI

### Machine Learning

- XGBoost
- scikit-learn
- pandas
- NumPy

### Database

- SQLite
- SQLAlchemy ORM

---

## Структура проекта

```text
.
├── db_functions/
│   └── db_manipulations.py
│
├── templates/
│   ├── base.html
│   ├── map.html
│   └── predict.html
│
├── data_fetcher.py
├── database.py
├── main.py
├── ml_model.py
├── run.py
├── vineyards_v2.db
└── .gitignore
```

---

## Описание основных файлов

### `main.py`

Главный файл FastAPI-приложения.

Содержит:

- инициализацию FastAPI;
- подключение Jinja2-шаблонов;
- frontend-маршруты;
- backend API;
- обработку координат;
- вызов сбора данных;
- вызов ML-модели;
- сохранение результата в базу.

Основные маршруты:

```text
GET  /
GET  /predict_page
GET  /api/vineyards
POST /api/predict
```

---

### `run.py`

Файл для локального запуска приложения.

Он:

- импортирует объект `app` из `main.py`;
- запускает Uvicorn;
- открывает приложение в браузере;
- использует хост `127.0.0.1`;
- использует порт `5459`.

После запуска приложение доступно по адресу:

```text
http://127.0.0.1:5459
```

---

### `database.py`

Файл с настройкой базы данных.

Содержит:

- подключение к SQLite;
- создание SQLAlchemy engine;
- создание сессий;
- ORM-модель `VineyardFeature`;
- автоматическое создание таблиц.

Используемая база:

```text
vineyards_v2.db
```

Основная таблица:

```text
vineyard_features
```

---

### `data_fetcher.py`

Модуль получения экологических данных по координатам.

При активном Google Earth Engine используются:

- `USGS/SRTMGL1_003` для данных рельефа;
- `ee.Terrain.products()` для уклона, экспозиции и hillshade;
- `IDAHO_EPSCOR/TERRACLIMATE` для климатических данных;
- `LANDSAT/LC08/C02/T1_L2` для расчёта NDVI и NDWI.

Если Google Earth Engine недоступен, возвращается fallback-набор данных с нулевыми значениями и статусом `FAILED`.

---

### `ml_model.py`

Модуль машинного обучения.

Отвечает за:

- обучение модели;
- загрузку сохранённой модели;
- подготовку признаков;
- выполнение прогноза пригодности участка.

Используемый алгоритм:

```text
XGBoost Classifier
```

Файл модели:

```text
xgboost_grape_model.json
```

---

### `db_functions/db_manipulations.py`

Вспомогательный скрипт для работы с базой данных.

Предназначен для миграции, обновления и обогащения данных экологическими признаками.

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SArtemA/vitisai_2sem.git
cd vitisai_2sem
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
```

### 3. Активация виртуального окружения

Для Windows:

```bash
venv\Scripts\activate
```

Для Linux / macOS:

```bash
source venv/bin/activate
```

### 4. Установка зависимостей

В репозитории нет отдельного файла `requirements.txt`, поэтому зависимости можно установить вручную:

```bash
pip install fastapi uvicorn sqlalchemy jinja2 pydantic xgboost pandas numpy scikit-learn earthengine-api requests
```

После установки рекомендуется создать `requirements.txt`:

```bash
pip freeze > requirements.txt
```

### 5. Запуск приложения

```bash
python run.py
```

После запуска откройте в браузере:

```text
http://127.0.0.1:5459
```

При запуске через `run.py` браузер должен открыться автоматически.

---

## Настройка Google Earth Engine

Проект может использовать Google Earth Engine для получения реальных геопространственных данных.

Перед использованием необходимо установить и авторизовать Earth Engine CLI:

```bash
pip install earthengine-api
earthengine authenticate
```

В коде используется проект:

```python
ee.Initialize(project='pp-2-sem-grapes')
```

Если у вас другой Google Cloud / Earth Engine project, замените значение `project` в `data_fetcher.py`.

Пример:

```python
ee.Initialize(project='your-project-id')
```

Если Google Earth Engine не настроен, приложение всё равно запустится, но данные будут получены из fallback-функции. В таком случае признаки будут заполнены нулями, а статусы будут иметь значение `FAILED`.

---

## API

### Получение списка виноградников

```http
GET /api/vineyards
```

Endpoint возвращает все записи из таблицы `vineyard_features`.

Пример ответа:

```json
[
  {
    "osm_id": 1,
    "lat": 44.6167,
    "lon": 33.5254,
    "elevation_GEE_USGS_30m": 120.0,
    "slope_GEE_USGS_30m": 8.2,
    "aspect_GEE_USGS_30m": 180.0,
    "hillshade_GEE_USGS_30m": 210.0,
    "mid_year_temp": 25.1,
    "precipitation": 420.0,
    "ndvi": 0.61,
    "ndwi": 0.18,
    "is_suitable": true
  }
]
```

---

### Прогноз пригодности участка

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
    "elevation_status": "GEE_SUCCESS",
    "slope": 8.2,
    "slope_status": "GEE_SUCCESS",
    "aspect": 180.0,
    "aspect_status": "GEE_SUCCESS",
    "hillshade": 210.0,
    "hillshade_status": "GEE_SUCCESS",
    "mid_year_temp": 25.1,
    "precipitation": 420.0,
    "ndvi": 0.61,
    "ndwi": 0.18
  }
}
```

---

## База данных

Проект использует SQLite-базу:

```text
vineyards_v2.db
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
| `elevation_GEE_USGS_30m` | Float | Высота над уровнем моря |
| `elevation_GEE_USGS_30m_status` | String | Статус получения высоты |
| `slope_GEE_USGS_30m` | Float | Уклон поверхности |
| `slope_GEE_USGS_30m_status` | String | Статус получения уклона |
| `aspect_GEE_USGS_30m` | Float | Экспозиция склона |
| `aspect_GEE_USGS_30m_status` | String | Статус получения экспозиции |
| `hillshade_GEE_USGS_30m` | Float | Освещённость рельефа |
| `hillshade_GEE_USGS_30m_status` | String | Статус получения hillshade |
| `mid_year_temp` | Float | Средняя / сезонная температура |
| `precipitation` | Float | Количество осадков |
| `ndvi` | Float | Индекс растительности |
| `ndwi` | Float | Индекс влажности |
| `is_suitable` | Boolean | Целевая переменная пригодности |

---

## ML-модель

Для классификации используется `XGBClassifier`.

Модель решает бинарную задачу:

```text
0 — участок не подходит
1 — участок подходит
```

### Используемые признаки

```text
elevation
slope
aspect
hillshade
mid_year_temp
precipitation
ndvi
ndwi
```

### Целевая переменная

```text
is_suitable
```

### Обучение

Модель обучается на данных из таблицы `vineyard_features`.

При обучении выполняются следующие шаги:

1. чтение данных из SQLite;
2. выбор нужных признаков;
3. очистка и преобразование данных;
4. заполнение пропусков нулями;
5. разделение выборки на train/test;
6. обучение XGBoost;
7. подбор гиперпараметров через `GridSearchCV`;
8. вывод accuracy, classification report и confusion matrix;
9. сохранение модели в файл.

Файл обученной модели:

```text
xgboost_grape_model.json
```

### Ручное обучение модели

Можно запустить обучение из Python:

```python
import ml_model

ml_model.train_model()
```

Или временно добавить вызов в отдельный скрипт:

```python
from ml_model import train_model

train_model()
```

---

## Миграция и обогащение данных

Для переноса данных и добавления экологических признаков используется скрипт:

```bash
python db_functions/db_manipulations.py
```

Сценарий работы скрипта:

1. читает существующие координаты;
2. получает дополнительные признаки;
3. формирует обновлённый набор данных;
4. сохраняет результат в новую базу.

---

## Настройки проекта

Сейчас ключевые настройки находятся прямо в коде.

### Хост и порт

Файл:

```text
run.py
```

```python
_HOST = '127.0.0.1'
_PORT = 5459
```

### База данных

Файл:

```text
database.py
```

```python
SQLALCHEMY_DATABASE_URL = "sqlite:///./vineyards_v2.db"
```

### Путь к модели

Файл:

```text
ml_model.py
```

```python
MODEL_PATH = "xgboost_grape_model.json"
DB_PATH = "vineyards_v2.db"
```

### Google Earth Engine project

Файл:

```text
data_fetcher.py
```

```python
ee.Initialize(project='pp-2-sem-grapes')
```

---


### Google Earth Engine не активен

В консоли может появиться сообщение:

```text
GEE Not Active
```

Возможные причины:

- не выполнена авторизация;
- нет доступа к проекту Google Earth Engine;
- указан неверный `project`;
- отсутствует интернет-соединение.

Решение:

```bash
earthengine authenticate
```

Также проверьте значение `project` в `data_fetcher.py`.

---

### Модель не найдена

Если файла `xgboost_grape_model.json` нет, приложение попытается обучить модель автоматически при прогнозе.

Возможные причины ошибки:

- отсутствует `vineyards_v2.db`;
- в базе недостаточно данных;
- в таблице нет нужных колонок;
- в целевой переменной только один класс.

Решение:

1. проверьте наличие базы данных;
2. проверьте таблицу `vineyard_features`;
3. убедитесь, что есть колонка `is_suitable`;
4. запустите обучение вручную.

---

### Ошибка при запуске FastAPI

Проверьте, что установлены зависимости:

```bash
pip install fastapi uvicorn sqlalchemy jinja2 pydantic
```

---

### Ошибка XGBoost

Проверьте установку:

```bash
pip install xgboost
```

Если ошибка связана с данными, проверьте, что признаки можно привести к числовому типу.

---

### Прогноз всегда возвращает одинаковый результат

Возможные причины:

- fallback-режим возвращает только нули;
- модель обучена на несбалансированных данных;
- в обучающей выборке мало примеров;
- признаки не имеют достаточной вариативности.

---

## Пример полного сценария запуска

```bash
git clone https://github.com/SArtemA/vitisai_2sem.git
cd vitisai_2sem

python -m venv venv
venv\Scripts\activate

pip install fastapi uvicorn sqlalchemy jinja2 pydantic xgboost pandas numpy scikit-learn earthengine-api requests

earthengine authenticate

python run.py
```

После запуска:

```text
http://127.0.0.1:5459
```

---

## Автор

**Дописать имена**

GitHub: [https://github.com/SArtemA](https://github.com/SArtemA)

---

## Лицензия

А НУЖНА ЛИ ???

Лицензия в репозитории не указана.

Перед использованием проекта в коммерческих или публичных целях рекомендуется добавить файл `LICENSE`.

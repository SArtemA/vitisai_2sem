import ee
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))

# Test
a = ee.Number(5)
b = ee.Number(10)
c = a.add(b)

try:
    result = c.getInfo()
    print(f"Результат вычисления 5 + 10 в GEE: {result}")

    dem = ee.Image('USGS/SRTMGL1_003')
    dem_info = dem.getInfo()
    print(f"Подключение к базе данных SRTM прошло успешно. ID первого тайла: {dem_info['id']}")

except Exception as e:
    print(f"Ошибка при выполнении запроса к серверу: {e}")
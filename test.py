"""
pp-2-sem-grapes

"""
from ml_model import *



# import sqlite3
# import pandas as pd
#
# # Update the path if your database file has a different name
# db_path = "vineyards_v2.db"
#
#
# def print_first_rows():
#     conn = sqlite3.connect(db_path)
#
#     # Query the first 5 rows
#     query = "SELECT * FROM vineyard_features LIMIT 5"
#     df = pd.read_sql_query(query, conn)
#
#     # Print the table
#     print(df.to_string())
#
#     conn.close()
#
#
# if __name__ == "__main__":
#     print_first_rows()


# import ee
# ee.Initialize(project='pp-2-sem-grapes')
# # Try to get the elevation of Mt. Everest
# dem = ee.Image('USGS/SRTMGL1_003')
# xy = ee.Geometry.Point([86.92, 27.98])
# res = dem.sample(xy, 30).first().getInfo()
# print(f"Elevation of Everest: {res['properties']['elevation']}m")
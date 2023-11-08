import geopandas as gpd
import pandas as pd

inf_count = pd.read_csv('urban_infrastructure_count.csv')
inf_count = inf_count[['CityCode','EntityCount']].groupby('CityCode').sum().reset_index()
selected_city = gpd.read_parquet('./city_information.parquet')
selected_city['geometry'] = selected_city['geometry'].centroid
selected_city = selected_city.merge(inf_count, on='CityCode',how='left')
selected_city['EntityCountNorm'] = selected_city['EntityCount']/selected_city['Area']
selected_city.to_file('./city_center.shp', driver='ESRI Shapefile')
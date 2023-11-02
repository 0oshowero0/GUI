import geopandas as gpd

selected_city = gpd.read_parquet('./city_information.parquet')
selected_city['geometry'] = selected_city['geometry'].centroid

selected_city.to_file('./city_center.shp', driver='ESRI Shapefile')
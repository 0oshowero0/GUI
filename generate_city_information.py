import os
os.environ['USE_PYGEOS'] = '0'
from shapely.geometry import Point, Polygon
import geopandas as gpd
import pandas as pd
from tqdm import tqdm


esri_worldcity = gpd.read_file('./Data/World_Cities.geojson',driver='GeoJSON') # 读取Esri WorldCity数据
gub_2018 = gpd.read_file('./Data/GUB_Global_2018/GUB_Global_2018.shp') # 读取GUB_2018数据

all_city_city_name_list = []  # Esri给出的城市名
all_city_province_name_list = []  # Esri给出的省名
all_city_country_code = [] # Esri给出的国家编码
all_city_country_name = []# Esri给出的国家名称
all_city_city_status = [] # Esri给出的城市状态
all_city_city_pop = [] # Esri给出的城市人口
all_city_city_pop_class = [] # Esri给出的城市人口分类
all_city_geometry_list = [] # geometry列表
all_city_id_list = [] # id
all_city_area_list = [] # 面积


idx = 0
for _, row in tqdm(esri_worldcity.iterrows(), total=esri_worldcity.shape[0]):
    # 根据Esri WorldCity的城市中心点，找到对应的GUB_2018的城市边界
    gub_2018['match'] = gub_2018['geometry'].apply(lambda x: Point((row['geometry'].x, row['geometry'].y)).within(x))
    gdf = gub_2018.loc[gub_2018['match']==True].reset_index(drop=True)
    if gdf.shape[0] == 0:
        continue
    else:
        geometry = gdf.loc[0]['geometry']
        area = gdf.loc[0]['urbanArea']
        id = 'A' + str(idx).zfill(5)
        all_city_city_name_list.append(row['CITY_NAME'])
        all_city_province_name_list.append(row['ADMIN_NAME'])
        all_city_country_code.append(row['FIPS_CNTRY'])
        all_city_country_name.append(row['CNTRY_NAME'])
        all_city_city_status.append(row['STATUS'])
        all_city_city_pop.append(row['POP'])
        all_city_city_pop_class.append(row['POP_CLASS'])
        all_city_geometry_list.append(geometry)
        all_city_id_list.append(id)
        all_city_area_list.append(area)

        idx += 1



all_city_gpd = gpd.GeoDataFrame({'CityName': all_city_city_name_list, 'StateName': all_city_province_name_list, 
                                 'CountryCode': all_city_country_code, 'CountryName': all_city_country_name,    
                                 'CityStatus': all_city_city_status, 'geometry': all_city_geometry_list, 
                                 'Population': all_city_city_pop, 'PopulationClass': all_city_city_pop_class,
                                 'ID': all_city_id_list, 
                                 'area': all_city_area_list}).set_geometry('geometry').set_crs(epsg=4326)

##############################################################################################################################
# 与世界银行数据合并

# 读取世界银行数据：GDP
gdp = pd.read_csv('./Data/World_Bank/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5871885/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5871885.csv')[['Country Name','Country Code','2020']].rename(columns={'Country Name':'CountryName','Country Code':'CountryCode','2020':'GDP'})
gdp = gdp.dropna()
# 读取世界银行数据：GDP Country Info
country_info = pd.read_csv('./Data/World_Bank/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5871885/Metadata_Country_API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5871885.csv')[['Country Code', 'Region', 'IncomeGroup']].rename(columns={'Country Code':'CountryCode'})
gdp = gdp.merge(country_info, on='CountryCode', how='outer')

# 读取世界银行数据：Per Capita GDP
percapitagdp = pd.read_csv('./Data/World_Bank/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5871588/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5871588.csv')[['Country Name','Country Code','2020']].rename(columns={'Country Name':'CountryName','Country Code':'CountryCode','2020':'PerCapitaGDP'})
percapitagdp = percapitagdp.dropna()
# 读取世界银行数据：Per Capita GDP Country Info
country_info = pd.read_csv('./Data/World_Bank/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5871588/Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5871588.csv')[['Country Code', 'Region', 'IncomeGroup']].rename(columns={'Country Code':'CountryCode'})
percapitagdp = percapitagdp.merge(country_info, on='CountryCode', how='left')

all_gdp = gdp.merge(percapitagdp, on=['CountryCode','CountryName','Region','IncomeGroup'], how='outer')


# 世界银行采用3位国家编码，而Esri WorldCity采用2位国家编码，需要转换
country_code_lut = pd.read_csv('./Data/country_code_lut.csv')[['Alpha2Mod','Alpha3']]
all_city_gpd = all_city_gpd.merge(country_code_lut, left_on='CountryCode', right_on='Alpha2Mod', how='left')

# 与gdp数据合并
all_city_gpdA = all_city_gpd.drop(['CountryCode', 'CountryName'], axis=1).merge(all_gdp, left_on='Alpha3', right_on='CountryCode', how='inner')
all_city_gpdB = all_city_gpd.drop(['CountryCode'], axis=1).merge(all_gdp, left_on='CountryName', right_on='CountryName', how='inner')

# 将两个结果合并
all_city_gpd_merge = pd.concat([all_city_gpdA, all_city_gpdB]).drop_duplicates(subset=['ID'], keep='last').reset_index(drop=True)
# 比较和all_city_gpd的差异，找到没有匹配上的城市；手动调整
# all_city_gpd['match'] = all_city_gpd['ID'].isin(all_city_gpd_merge['ID'])
# all_city_gpd_no_match = all_city_gpd.loc[all_city_gpd['match']==False].reset_index(drop=True)
# print(all_city_gpd_no_match.drop_duplicates(subset=['CountryName'], keep='last').reset_index(drop=True))

all_city_gpd_merge = all_city_gpd_merge.drop(['Alpha2Mod','Alpha3','area'], axis=1).sort_values(by=['ID']).reset_index(drop=True)
all_city_gpd = all_city_gpd_merge



# 为了测量面积，可以重新投射到Lambert Cylindrical投影（epsg:6933）
# 兰伯特圆柱投影是一种保留面积测量的等面积投影。在地球表面具有相同大小的区域在地图上具有相同的大小。然而，形状、角度和比例尺一般会被扭曲。
all_city_gpd = all_city_gpd.to_crs(epsg=6933)
all_city_gpd['Area'] = all_city_gpd.geometry.area
all_city_gpd = all_city_gpd.to_crs(epsg=4326)

# all_city_gpd.to_file('all_city_information.geojson', driver='GeoJSON')  # 可用QGIS打开并选择合适的边界
# all_city_gpd.to_parquet('all_city_information.parquet') # parquet格式适合大量数据，可以快速读取
# all_city_gpd.drop(['geometry'], axis=1).to_csv('all_city.csv',sep=',', encoding='utf-8-sig',index=False) # 保存成csv可以简单看看

# 根据面积筛掉过大的城市
all_city_gpd_area_select = all_city_gpd.loc[all_city_gpd.Area<20000**2].reset_index(drop=True)
all_city_gpd_area_select.to_file('city_information.geojson', driver='GeoJSON')  # 可用QGIS打开并选择合适的边界
all_city_gpd_area_select.to_parquet('city_information.parquet') # parquet格式适合大量数据，可以快速读取


# 后处理:Debug
all_city_gpd_area_select = gpd.read_parquet('city_information.parquet')
all_city_gpd_area_select.loc[ all_city_gpd_area_select['CountryName'].isna()]
all_city_gpd_area_select.loc[(all_city_gpd_area_select['CountryCode']=='VEN'),'CountryName'] = 'Venezuela'
all_city_gpd_area_select.loc[(all_city_gpd_area_select['CountryCode']=='SSD'),'CountryName'] = 'South Sudan'
all_city_gpd_area_select.loc[(all_city_gpd_area_select['CountryCode']=='YEM'),'CountryName'] = 'Yemen'


# 生成最终数据集
all_city_gpd_area_select['CityCode'] = all_city_gpd_area_select['ID']
all_city_gpd_area_select = all_city_gpd_area_select[['CityCode','CityName','CountryCode','CountryName','Region','CityStatus','Population',
                                                        'PopulationClass','GDP','PerCapitaGDP','IncomeGroup','Area','geometry']]



all_city_gpd_area_select.to_file('city_information.geojson', driver='GeoJSON')  # 可用QGIS打开并选择合适的边界
all_city_gpd_area_select.to_parquet('city_information.parquet') # parquet格式适合大量数据，可以快速读取

# 去除geometry
all_city_gpd_area_select.drop(['geometry'], axis=1).to_csv('city_information.csv',sep=',', encoding='utf-8-sig',index=False) 
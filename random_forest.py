import numpy as np
from math import floor
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13


MULTI_PROCESS_TRAINING = 20
############################################################################################
# 数据路径
CITY_INFO_LOC = Path('./city_information.csv')
URBAN_INFRASTRUCTURE_COUNT_LOC = Path('./urban_infrastructure_count.csv')
# 读取数据
city_info = pd.read_csv(CITY_INFO_LOC)
urban_infrastructure_count = pd.read_csv(URBAN_INFRASTRUCTURE_COUNT_LOC)

# 聚合二者信息
infrastructure = city_info.merge(urban_infrastructure_count.drop(['CityName'],axis=1), on='CityCode', how='left')
infrastructure = infrastructure.dropna()
# 对每个国家，统计各个城市各个基础设施的平均值、中位数、总数
mean_inf = infrastructure.pivot(index=['CityCode','CountryCode'], columns='SubCategory', values='EntityCount').fillna(0).reset_index().groupby('CountryCode').mean().reset_index(drop=False)
median_inf = infrastructure.pivot(index=['CityCode','CountryCode'], columns='SubCategory', values='EntityCount').fillna(0).reset_index().groupby('CountryCode').median().reset_index(drop=False)
sum_inf = infrastructure.pivot(index=['CityCode','CountryCode'], columns='SubCategory', values='EntityCount').fillna(0).reset_index().groupby('CountryCode').sum().reset_index(drop=False)

features = mean_inf.merge(median_inf, on='CountryCode', how='left', suffixes=('_mean','_median')).merge(sum_inf, on='CountryCode', how='left', suffixes=('','_sum'))
features_with_label = features.merge(city_info[['CountryCode','GDP']].drop_duplicates(), on='CountryCode', how='left')
#features_with_label['GDP'] = (features_with_label['GDP'] - features_with_label['GDP'].mean()) / features_with_label['GDP'].std() 

train_index = np.random.choice(features_with_label.index, floor(len(features_with_label)*0.8), replace=False)
train_X = features_with_label.iloc[train_index,1:-1]
train_Y = features_with_label.iloc[train_index,-1]
test_X = features_with_label.drop(train_index, axis=0).iloc[:,1:-1]
test_Y = features_with_label.drop(train_index, axis=0).iloc[:,-1]

print('=================================================')
print('Fitting Start')
begin_time = datetime.now()
repeat_train = 5
R2_list = []
explained_variance_list = []

for i in range(repeat_train):
    clf = RandomForestRegressor(n_estimators=200, criterion='squared_error',max_depth=None, n_jobs=MULTI_PROCESS_TRAINING)
    clf.fit(train_X.to_numpy(), train_Y.to_numpy().reshape(-1))
    end_time = datetime.now()
    print('Time Consumption：' + str((end_time - begin_time).total_seconds() / 60) + ' minutes')
    pred = clf.predict(test_X)
    r2 = r2_score(test_Y.to_numpy().reshape(-1), pred)
    ev = explained_variance_score(test_Y.to_numpy().reshape(-1), pred)

    R2_list.append(r2)
    explained_variance_list.append(ev)

    print('=================================================')
    print(f'Test R2: {r2}, Test Explained Variance: {ev}')

df = pd.DataFrame({'R2':R2_list,'Explained Variance':explained_variance_list})
print('=================================================')
print('Average Result')
print(df.mean())

########################################################################################################







########################################################################################################
# Visualize
clf = RandomForestRegressor(n_estimators=20, criterion='squared_error',max_depth=None, n_jobs=MULTI_PROCESS_TRAINING)
clf.fit(train_X.to_numpy(), train_Y.to_numpy().reshape(-1))

fig, axes = plt.subplots(figsize=(8, 6))
importance_df = pd.DataFrame({'Feature Name':test_X.columns,'Importance':clf.feature_importances_})
# 将Feature Name按照_前进行聚合，加在一起
importance_df['Feature Name'] = importance_df['Feature Name'].apply(lambda x: x.split('_')[0])
importance_df = importance_df.groupby('Feature Name')['Importance'].agg('sum').sort_values(ascending=False).reset_index()
grp_order = list(importance_df.groupby('Feature Name')['Importance'].agg('mean').sort_values(ascending=False).index)
sns.barplot(data=importance_df,x='Feature Name',y='Importance',axes=axes,order=grp_order)
axes.set_ylabel("Feature Importance",size=24)
axes.set_xlabel("", size=20)
axes.tick_params(axis='y', labelsize=24)
axes.tick_params(axis='x', labelsize=20,rotation=90)
plt.grid()
plt.tight_layout()
plt.show()
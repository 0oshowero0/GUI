import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

sns.set_theme(style="white", palette=None)
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['xtick.labelsize']=13
plt.rcParams['ytick.labelsize']=13

#COLOR_LIST = sns.color_palette("Set2")
COLOR_LIST = ['#67ACFF','#3DB901','#E9A521','#FF5500']

############################################################################################
# 数据路径
CITY_INFO_LOC = Path('./city_information.csv')
URBAN_INFRASTRUCTURE_COUNT_LOC = Path('./urban_infrastructure_count.csv')
# 读取数据
city_info = pd.read_csv(CITY_INFO_LOC)
urban_infrastructure_count = pd.read_csv(URBAN_INFRASTRUCTURE_COUNT_LOC)

# 聚合二者信息
infrastructure = city_info.merge(urban_infrastructure_count.drop(['CityName'],axis=1), on='CityCode', how='left')

infrastructure_agg = infrastructure.groupby(['CityCode','CityName','IncomeGroup','Area','Population']).sum().reset_index()
infrastructure_agg = infrastructure_agg.dropna()
infrastructure_agg = infrastructure_agg.loc[infrastructure_agg['Population'] > 0].reset_index(drop=True)
infrastructure_agg['PopulationLog'] = np.log10(infrastructure_agg['Population'])
infrastructure_agg['AreaLog'] = np.log10(infrastructure_agg['Area']/1000/1000)
infrastructure_agg['EntityCountLog'] = np.log10(infrastructure_agg['EntityCount'])


############################################################################################

# 回归分析
def draw_single_dim(model_result, x_name, y_name):
    x = sm.add_constant(np.linspace(infrastructure_agg[x_name].min(),
                                    infrastructure_agg[x_name].max(), 100))
    pred_ols = model_result.get_prediction(x)
    iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(infrastructure_agg[x_name], infrastructure_agg[y_name], "o",
            label="data")
    ax.plot(x[:, 1], pred_ols.predicted, "r--.", label="OLS")
    ax.plot(x[:, 1], iv_u, "r--")
    ax.plot(x[:, 1], iv_l, "r--")
    ax.legend(loc="best", fontsize=16)
    ax.set_ylabel(y_name, size=24)
    ax.set_xlabel(x_name, size=24)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=16)
    plt.show()


def draw_single_dim_decomp(model_result, x_name, y_name, group_var):
    common_slope = model_result.params[x_name]
    common_intercept = model_result.params['Intercept']
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(infrastructure_agg[x_name], infrastructure_agg[y_name], ".", color='#7c7c7c')
    hic = infrastructure_agg.loc[infrastructure_agg[group_var] == 'High income']
    umic = infrastructure_agg.loc[infrastructure_agg[group_var] == 'Upper middle income']
    lmic = infrastructure_agg.loc[infrastructure_agg[group_var] == 'Lower middle income']
    lic = infrastructure_agg.loc[infrastructure_agg[group_var] == 'Low income']
    ax.plot(hic[x_name], hic[y_name], ".", color=colors.to_hex(COLOR_LIST[0]))
    ax.plot(umic[x_name], umic[y_name], ".", color=colors.to_hex(COLOR_LIST[1]))
    ax.plot(lmic[x_name], lmic[y_name], ".", color=colors.to_hex(COLOR_LIST[2]))
    ax.plot(lic[x_name], lic[y_name], ".",color=colors.to_hex(COLOR_LIST[3]))
    # ax.plot(infrastructure_agg[x_name], model_result.predict(), "k-")
    random_effects = model_result.random_effects
    x = np.linspace(infrastructure_agg[x_name].min(), infrastructure_agg[x_name].max(), 100)

    ax.plot(x, x * common_slope + common_intercept + x * random_effects['High income'][x_name] + random_effects['High income'][group_var], "--", color=colors.to_hex(COLOR_LIST[0]), label="HIC")
    ax.plot(x, x * common_slope + common_intercept + x * random_effects['Upper middle income'][x_name] + random_effects['Upper middle income'][group_var], "--", color=colors.to_hex(COLOR_LIST[1]), label="UMIC")
    ax.plot(x, x * common_slope + common_intercept + x * random_effects['Lower middle income'][x_name] + random_effects['Lower middle income'][group_var], "--", color=colors.to_hex(COLOR_LIST[2]), label="LMIC")
    ax.plot(x, x * common_slope + common_intercept + x * random_effects['Low income'][x_name] + random_effects['Low income'][group_var], "--", color=colors.to_hex(COLOR_LIST[3]), label="LIC")

    ax.legend(loc="best", fontsize=16)
    # ax.set_ylabel(y_name, size=24)
    # ax.set_xlabel(x_name, size=24)
    ax.set_ylabel('Entity Count', size=24)
    ax.set_xlabel('Population', size=24)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ori_y_ticks = ax.get_yticks()
    new_y_ticks = ["$10^{%(i).1f}$"%{'i':i} for i in ori_y_ticks]
    ax.set_yticks(ori_y_ticks,new_y_ticks)
    ori_x_ticks = ax.get_xticks()
    new_x_ticks = ["$10^{%(i).1f}$"%{'i':i} for i in ori_x_ticks]
    ax.set_xticks(ori_x_ticks,new_x_ticks)
    plt.tight_layout()
    plt.show()


####################################################################
# model0 = sm.OLS(infrastructure_agg['EntityCount'], sm.add_constant(infrastructure_agg[['Population']]))
# results0 = model0.fit()
# results0.summary()
# draw_single_dim(results0, 'EntityCount', 'Population')

# 构建随机斜率模型
model01 = sm.MixedLM.from_formula('EntityCountLog ~ PopulationLog', groups='IncomeGroup', re_formula="1+PopulationLog", data=infrastructure_agg)
results01 = model01.fit()
results01.summary()
# 取两个极端组拟合参数绘图
draw_single_dim_decomp(results01, 'PopulationLog', 'EntityCountLog', 'IncomeGroup')
print('随机效应截距'+str([round(results01.random_effects[i]['IncomeGroup'], 2) for i in ['High income','Upper middle income','Lower middle income','Low income']]))
print('随机效应斜率'+str([round(results01.random_effects[i]['PopulationLog'], 2) for i in ['High income','Upper middle income','Lower middle income','Low income']]))

# # 构建随机斜率模型
# model01 = sm.MixedLM.from_formula('EntityCountLog ~ AreaLog', groups='IncomeGroup', re_formula="1+AreaLog", data=infrastructure_agg)
# results01 = model01.fit()
# results01.summary()
# # 取两个极端组拟合参数绘图
# draw_single_dim_decomp(results01, 'AreaLog', 'EntityCountLog', 'IncomeGroup')
# print('随机效应截距'+str([round(results01.random_effects[i]['IncomeGroup'], 2) for i in ['High income','Upper middle income','Lower middle income','Low income']]))
# print('随机效应斜率'+str([round(results01.random_effects[i]['AreaLog'], 2) for i in ['High income','Upper middle income','Lower middle income','Low income']]))

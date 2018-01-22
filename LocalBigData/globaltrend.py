import pandas as pd




# aqi = pd.read_csv('C:\\Users\HP-LAPTOP\Downloads\daily_aqi_by_county_2016.csv')
# pred_aqi = pd.read_csv("C:\\Users\HP-LAPTOP\Downloads\predicted_aqi.csv")
#
# pred_aqi = pred_aqi[['month','year','global_aqi','state_code']]
# aqi = aqi[['Date','State Code','AQI']]
# date = pd.DataFrame(aqi['Date'].str.split('-').tolist())
#
# aqi['year'] = date[0].astype(int)
# aqi['month'] = date[1].astype(int)
# aqi.drop(['Date'],axis = 1,inplace= True)
# aqi = aqi.reindex_axis(["month","year","AQI","State Code"],axis =1)
# aqi = aqi.groupby(['month','State Code'],as_index=False).agg({'AQI':'avg'})
# #print(aqi)
# pred_aqi.rename(columns={'state_code':'State Code'},inplace=True)
# aqi_plot = pd.merge(aqi, pred_aqi,left_on=['month','State Code'], right_on=['month','State Code'])
#
# aqi_plot["Diff"] = aqi_plot['global_aqi'] - aqi_plot['AQI']
# print(aqi_plot)
# aqi_plot.to_csv('C:\\Users\HP-LAPTOP\Downloads\diff_AQI.csv',header = True,index = False)
path = "C:\\Users\HP-LAPTOP\Downloads\daily_TEMP_2016.csv"
# path1 = "C:\\Users\HP-LAPTOP\Downloads\daily_PRESS_2016.csv"
# path2 = "C:\\Users\HP-LAPTOP\Downloads\daily_RH_DP_2016.csv"
# path3 = "C:\\Users\HP-LAPTOP\Downloads\daily_WIND_2016.csv"
temp = pd.read_csv(path)
# press = pd.read_csv(path1)
# rh = pd.read_csv(path2)
# wind = pd.read_csv(path3)

pred_temp = pd.read_csv("C:\\Users\HP-LAPTOP\Downloads\predicted_temp.csv")
# pred_press = pd.read_csv("C:\\Users\HP-LAPTOP\Downloads\predicted_press.csv")
# pred_rh = pd.read_csv("C:\\Users\HP-LAPTOP\Downloads\predicted_rh.csv")
# pred_wind = pd.read_csv("C:\\Users\HP-LAPTOP\Downloads\predicted_wind.csv")


temp = temp[['Date Local','State Code','Arithmetic Mean']]
date = pd.DataFrame(temp['Date Local'].str.split('-').tolist())

temp['year'] = date[0].astype(int)
temp['month'] = date[1].astype(int)
temp.drop(['Date Local'],axis = 1,inplace= True)
temp = temp.reindex_axis(["month","year","Arithmetic Mean","State Code"],axis =1)
temp = temp.groupby(['month','State Code'],as_index=False).mean()#.agg({'Arithmetic Mean':'avg'})
pred_temp.rename(columns={'state_code':'State Code'},inplace=True)
temp_plot = pd.merge(temp, pred_temp,left_on=['month','State Code'], right_on=['month','State Code'])

temp_plot["Diff"] = temp_plot['predicted_temp'] - temp_plot['Arithmetic Mean']

print(temp_plot)
# press = press[['Date Local','State Code','Arithmetic Mean']]
# date = pd.DataFrame(press['Date Local'].str.split('-').tolist())
#
# press['year'] = date[0].astype(int)
# press['month'] = date[1].astype(int)
# press.drop(['Date Local'],axis = 1,inplace= True)
# press = press.reindex_axis(["month","year","Arithmetic Mean","State Code"],axis =1)
# pred_press.rename(columns={'state_code':'State Code'},inplace=True)
# press_plot = pd.merge(press, pred_press,left_on=['month','State Code'], right_on=['month','State Code'])
#
# press_plot["Diff"] = press_plot['predicted_press'] - press_plot['Arithmetic Mean']
#
# rh = rh[['Date Local','State Code','Arithmetic Mean']]
# date = pd.DataFrame(rh['Date Local'].str.split('-').tolist())
#
# rh['year'] = date[0].astype(int)
# rh['month'] = date[1].astype(int)
# rh.drop(['Date Local'],axis = 1,inplace= True)
# rh = rh.reindex_axis(["month","year","Arithmetic Mean","State Code"],axis =1)
# pred_rh.rename(columns={'state_code':'State Code'},inplace=True)
# rh_plot = pd.merge(rh, pred_rh,left_on=['month','State Code'], right_on=['month','State Code'])
#
# rh_plot["Diff"] = rh_plot['predicted_rh'] - rh_plot['Arithmetic Mean']
#
# wind = wind[['Date Local','State Code','Arithmetic Mean']]
# date = pd.DataFrame(wind['Date Local'].str.split('-').tolist())
#
# wind['year'] = date[0].astype(int)
# wind['month'] = date[1].astype(int)
# wind.drop(['Date Local'],axis = 1,inplace= True)
# wind = wind.reindex_axis(["month","year","Arithmetic Mean","State Code"],axis =1)
# pred_wind.rename(columns={'state_code':'State Code'},inplace=True)
# wind_plot = pd.merge(wind, pred_wind,left_on=['month','State Code'], right_on=['month','State Code'])
#
# wind_plot["Diff"] = wind_plot['predicted_wind'] - wind_plot['Arithmetic Mean']
#
#
temp_plot.to_csv('C:\\Users\HP-LAPTOP\Downloads\diff_TEMP.csv',header = True,index = False)
# press_plot.to_csv('C:\\Users\HP-LAPTOP\Downloads\diff_PRESS.csv',header = True,index = False)
# rh_plot.to_csv('C:\\Users\HP-LAPTOP\Downloads\diff_RH.csv',header = True,index = False)
# wind_plot.to_csv('C:\\Users\HP-LAPTOP\Downloads\diff_WIND.csv',header = True,index = False)
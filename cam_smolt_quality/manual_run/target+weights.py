import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
from cam_smolt_quality.configs import READ_PARAMS
import cam_smolt_quality.manual_run.utils as utils

### 1. base/20221222 Smolt performance Phase 3_copy.ipynb
df2 = utils.read_file('eb_stocking_edited2', READ_PARAMS, buffer=True)
df1 = utils.read_file('mortality_target_to_be_grouped', READ_PARAMS, buffer=True) # data on first 90 days after transfer #e)
freshwater_names=utils.read_file('from_locus_name_lookup', READ_PARAMS, buffer=True)

df1.drop(columns=['event_date.1', 'locus_id.1', 'fish_group_id.1'],inplace=True)
df2.from_date=pd.to_datetime(df2.from_date,format='%Y-%m-%d')
df2.to_date=pd.to_datetime(df2.to_date,format='%Y-%m-%d')
df2.transfer_date=pd.to_datetime(df2.transfer_date,format='%Y-%m-%d')
df2['days_btw_to_from']=(df2.to_date-df2.from_date).dt.days
df2['days_btw_to_transfer']=(df2.to_date-df2.transfer_date).dt.days
weight_bins = np.linspace(100, 225, num=6)
df2['to_avg_weight_binned'] = pd.cut(df2['to_avg_weight'], weight_bins)

df1['total_mortality_perc_90']=df1['total_mortality']/df1['total_count']
df1['transport_mortality_perc_90']=df1['transport_mortality']/df1['total_count']
df1['nontransport_mortality_perc_90']=df1['nontransport_mortality']/df1['total_count']

df1.transfer_date=pd.to_datetime(df1.transfer_date,format='%Y-%m-%d')
df1['transfer_year']=df1.transfer_date.dt.year
df1['transfer_month']=df1.transfer_date.dt.month
df1['transfer_month_year']=df1['transfer_month'].astype(str)+'_'+df1['transfer_year'].astype(str)
season_dic = {1: 'winter',2: 'spring',3: 'summer',4: 'autumn'}
df1['transfer_season']=(df1['transfer_date'].dt.month%12 // 3 + 1).apply(lambda x: season_dic[x])
season_dic2 = {1: 'Dec-Feb',2: 'Mar-May',3: 'Jun-Aug',4: 'Sep-Nov'}
df1['transfer_season2']=(df1['transfer_date'].dt.month%12 // 3 + 1).apply(lambda x: season_dic2[x])
reverse_season_dic = {v: k for k, v in season_dic.items()}

df1.event_date=pd.to_datetime(df1.event_date,format='%Y-%m-%d')
mortality=df1.dropna().groupby(['locus_id','fish_group_id']).agg({'transfer_year':'min'
                                                                 ,'transfer_month':'min'
                                                                 ,'transfer_month_year':'min' 
                                                                 ,'transfer_season':'min'
                                                                 ,'transfer_season2':'min'
                                                                 ,'total_count':'mean'
                                                                 ,'total_mortality_perc_90':'sum'
                                                                 ,'transport_mortality_perc_90':'sum'
                                                                 ,'nontransport_mortality_perc_90':'sum'
                                                                 }).reset_index()
mortality = mortality[mortality.total_count <= mortality.total_count.quantile(.975)]
mortality = mortality[mortality.total_count > 10000]
mortality = mortality[mortality.transport_mortality_perc_90 < mortality.transport_mortality_perc_90.quantile(.995)]
mortality = mortality[mortality.total_mortality_perc_90 < mortality.total_mortality_perc_90.quantile(.99)]
mortality = mortality[mortality.nontransport_mortality_perc_90 < mortality.nontransport_mortality_perc_90.quantile(.99)]

df3 = mortality.merge(df2, how='left', left_on=['locus_id', 'fish_group_id'], right_on=['to_locus_id', 'to_fish_group_id'])
# df3 = mortality_grand.merge(df2, how='left', left_on=['locus_id', 'fish_group_id'], right_on=['to_locus_id', 'to_fish_group_id'])
print('Saving smolt_dataset_transfers.csv to data folder')
df3.drop(columns=['to_avg_weight_binned']).to_csv('cam_smolt_quality/data/smolt_dataset_transfers.csv',index=False) #_until2023Feb28_short



## 2.base/20230712 Target comparison_only_mortality_nSFR:
mortality=pd.read_csv('cam_smolt_quality/data/smolt_dataset_transfers.csv') #new/ _until2023May18_short
inv=utils.read_file('evt_inventory_only_SW_cages_only_since_2017', READ_PARAMS, buffer=True)

key_columns = ['locus_id','fish_group_id','transfer_year',] 
df=mortality[key_columns+['to_avg_weight','total_mortality_perc_90','transport_mortality_perc_90','nontransport_mortality_perc_90']]
df.rename(columns={'to_avg_weight':'stocking_weight'},inplace=False)

inv['open_biomass_kg']=inv['open_count']*inv['open_weight']/1000
inv['oSFR'] = np.where(inv['open_biomass_kg'] == 0, np.nan, inv['feed_amount'] / inv['open_biomass_kg'] * 100)
inv['eSFR'] = inv.apply(utils.eSFR,axis=1)
inv['nSFR'] = np.where(inv['eSFR'] == 0, np.nan, inv['oSFR'] / inv['eSFR'])

#creating new dataframe with 90 dates for each transfer
tmpp=mortality[key_columns+['transfer_date']]
tmpp['transfer_date']=pd.to_datetime(tmpp['transfer_date'], utc=True)
tmpp['sw90_date'] = tmpp['transfer_date'] + pd.Timedelta(90,'d')

# Apply the function to each row and concatenate the results
new_df = pd.concat(tmpp.apply(utils.generate_event_dates, axis=1).tolist(), ignore_index=True)

inv_grouped=inv.groupby(['event_date','locus_id'])[['oSFR','eSFR','nSFR']].max().reset_index()
df_daily = new_df.merge(mortality[key_columns+['transfer_date']]).merge(inv_grouped, how='left')
df_daily['transfer_date']=pd.to_datetime(df_daily['transfer_date'], utc=True)
df_daily=df_daily[df_daily.transfer_date < df_daily.event_date]
df_daily['nSFR'] = np.where(df_daily['eSFR'] < 0, np.nan, df_daily['nSFR'])
df_daily['oSFR'] = df_daily['oSFR'].fillna(0)

df=df.merge(df_daily.groupby(key_columns)[['oSFR','nSFR']].mean().reset_index())
df['log_mortality']=np.log(df['total_mortality_perc_90'])
df.to_csv('cam_smolt_quality/data/targets.csv',index=False)



## 3. base (dmitrii)/FW data processing (Dmitriis optimization).ipynb
RECALCULATE_WEIGHTS = True


locus_weights = utils.read_file('evt_movement_ratio_with_dates', READ_PARAMS, buffer=True)
temperature = utils.read_file('temperature_for_CAM', READ_PARAMS, buffer=True)
temperature.locus_group_id=temperature.locus_group_id.astype('int16')
#not sure in row below as it 'converts' 12.3 -> 12.296875
temperature.value=temperature.value.astype('float16')
temperature['event_year']=temperature['event_date'].dt.year

llg_match = utils.read_file('locus_locus_group_matching', READ_PARAMS, buffer=True)
llg_match.locus_id=llg_match.locus_id.astype('int32')
llg_match.locus_group_id=llg_match.locus_group_id.astype('int16')

df_dates = utils.read_file('FW_cycle_dates', READ_PARAMS, buffer=True)
    
sfm = utils.read_file('seawater_freshwater_matching', READ_PARAMS, buffer=True)
sfm_ = sfm[sfm.origin_site_type=='Freshwater'][['target_seawater_locus_id','transport_date','ponding_date','pretransfer_fw_locus_population_id','fish_count_shipped_out','avg_weight_g_stocked']]
sfm_.pretransfer_fw_locus_population_id=sfm_.pretransfer_fw_locus_population_id.astype('int64')

lw_dates=locus_weights.groupby('final_locus_population_id').agg({'starttime':'min','endtime':'max'})
lw_dates.starttime = pd.to_datetime(lw_dates.starttime,format='%Y-%m-%d')
lw_dates.endtime = pd.to_datetime(lw_dates.endtime,format='%Y-%m-%d')

#to be checked
lw_dates['FW_cycle_length'] = (lw_dates.endtime - lw_dates.starttime).dt.days+1
lw_dates['starttime_year']=lw_dates['starttime'].dt.year
#we limit FW cycles to those started in 2017 because there are issues with temperature readings for 2015-2016
lw_dates_2017=lw_dates[lw_dates.starttime_year>=2017]

N = len(locus_weights['final_locus_population_id'].unique())
rnd_idxs = np.random.randint(0, N, 50)
mask = locus_weights['final_locus_population_id'].unique()[rnd_idxs]
mask = locus_weights['final_locus_population_id'].isin(mask)

if RECALCULATE_WEIGHTS:

    print('Processing weights in locuses...')
    #time consuming
    lw_alldates_list = []
    for ind, row in tqdm(locus_weights[mask].iterrows()):
        for d in pd.date_range(row.starttime,row.endtime-datetime.timedelta(days=1)):
            lw_alldates_list.append([row.final_locus_population_id,d,row.historic_locus_id,row.count_ratio])
    lw_alldates = pd.DataFrame(lw_alldates_list, columns = ['final_locus_population_id','event_date','historic_locus_id','weight0'])
    lw_alldates_weights_grouped=lw_alldates.groupby(['final_locus_population_id','event_date'])[['weight0']].sum().reset_index()
    lw_alldates_weights_grouped_merged=lw_alldates.merge(lw_alldates_weights_grouped, on=['final_locus_population_id','event_date'], how='left')
    lw_alldates_weights_grouped_merged['weight']=lw_alldates_weights_grouped_merged['weight0_x']/lw_alldates_weights_grouped_merged['weight0_y']
    lw_alldates_final=lw_alldates_weights_grouped_merged[['final_locus_population_id', 'event_date', 'historic_locus_id','weight']].sort_values(by=['final_locus_population_id','event_date','historic_locus_id'])
    lw_alldates_final.historic_locus_id=lw_alldates_final.historic_locus_id.astype('int32')

    lw_alldates_final = []
    for lp_id, lp_df in tqdm(locus_weights.groupby('final_locus_population_id'), 'Assigning LP weights'):
        e_df = utils.expand_dates_vectorized(lp_df)
        # e_df = pd.concat([expand_dates(row) for _, row in lp_df.iterrows()], ignore_index=True)
        agg_df = e_df.groupby(['event_date', 'final_locus_population_id']).sum().reset_index()
        agg_df_merged=e_df.merge(agg_df, on=['final_locus_population_id','event_date', 'historic_locus_id'], how='left')
        agg_df_merged['weight']=agg_df_merged['weight0_x']/agg_df_merged['weight0_y']
        res_df = agg_df_merged[['final_locus_population_id', 'event_date', 'historic_locus_id','weight']].sort_values(by=['final_locus_population_id','event_date','historic_locus_id'])
        res_df.historic_locus_id=res_df.historic_locus_id.astype('int32')

        lw_alldates_final.append(res_df)
    lw_alldates_final = pd.concat(lw_alldates_final, axis=0)
    print('-- Saving lw_alldates_final')
    lw_alldates_final.to_csv('cam_smolt_quality/data/lw_alldates_final.csv', index=False)



## 4. temperature/20230607 FW data processing.ipynb
lw_alldates_final=utils.read_file('lw_alldates_final', buffer=True)

llg_match = dict(zip(llg_match['locus_id'], llg_match['locus_group_id']))
lw_alldates_final['locus_group_id'] = lw_alldates_final['historic_locus_id'].replace(llg_match)
lw_alldates_final.drop(columns='historic_locus_id',inplace=True)

lw_alldates_final=lw_alldates_final.merge(temperature[['locus_group_id', 'event_date', 'value']],how='left')
lw_alldates_final=lw_alldates_final[(lw_alldates_final['value'].notna())]
lw_alldates_final['event_year']=lw_alldates_final['event_date'].dt.year

lw_alldates_final.rename(columns={'value':'temperature'},inplace=True)
lw_alldates_final.temperature=lw_alldates_final.temperature.astype('float16').round(1)
lw_alldates_final['weight_temperature']=lw_alldates_final['weight']*lw_alldates_final['temperature']

dft=lw_alldates_final.groupby(['final_locus_population_id','event_date'])[['weight_temperature']].agg(lambda x: x.sum(skipna=False)).reset_index()
del(lw_alldates_final)
dft.rename(columns={'weight_temperature':'temperature'},inplace=True)
dft['temperature']=dft['temperature'].round(1).astype('str')


df_dates_2017=df_dates.merge(lw_dates_2017.reset_index()[['final_locus_population_id']],left_on='pretransfer_fw_locus_population_id',right_on='final_locus_population_id',how='inner')
df_dates_2017.drop(columns=['final_locus_population_id'],inplace=True)

tmp_list=[]
for ind,row in df_dates_2017.iterrows():
    lp = row.pretransfer_fw_locus_population_id
    start = row.first_movement_date
    end = row.shipout_date
    for d in pd.date_range(start,end):
        tmp_list.append([lp,d])
tmp_df=pd.DataFrame(tmp_list,columns=['final_locus_population_id','event_date'])
dft_=tmp_df.merge(dft,how='left')
#interpolation method #1 without handling outliers
output_df_temp = pd.DataFrame()
for ind,curr_df in dft_.groupby('final_locus_population_id'):
    tmp_df=curr_df.copy()
    tmp_df.temperature=curr_df.temperature.interpolate()
    output_df_temp=pd.concat([output_df_temp,tmp_df])
dft_filled = output_df_temp.copy()
dft_filled.to_csv('cam_smolt_quality/data/FW_temperature_filled.csv',index=False)
del(dft_, dft_filled, locus_weights, output_df_temp)


## 5. temperature/FW_temperature_cleared.ipynb
df_date_temperature=utils.read_file('FW_temperature_filled',buffer=True)
# Drop NaN values
df_date_temperature = df_date_temperature.dropna()

# Get unique part numbers
unic_part_number_list = df_date_temperature['final_locus_population_id'].unique()

# Calculate rolling mean for the entire DataFrame
df_date_temperature['rolling_tempr'] = df_date_temperature.groupby('final_locus_population_id')['temperature'].rolling(30, min_periods=1).mean().reset_index(level=0, drop=True)

# Calculate the mean of the first 30 temperatures for each part number
average_tempr_30 = df_date_temperature.groupby('final_locus_population_id')['temperature'].apply(lambda x: x.head(30).mean())
half_average_tempr_30 = average_tempr_30 / 2

# Apply the np.where condition
for part_number in tqdm(unic_part_number_list):
    first_30_temps = df_date_temperature.loc[df_date_temperature['final_locus_population_id'] == part_number, 'temperature'].head(30)
    t_dif = first_30_temps - average_tempr_30[part_number]
    condition_indices = first_30_temps.index
    df_date_temperature.loc[condition_indices, 'rolling_tempr'] = np.where(abs(t_dif) < half_average_tempr_30[part_number], first_30_temps, average_tempr_30[part_number])

# Your final DataFrame
df_temperature_cleared = df_date_temperature

# Assuming df_temperature_cleared is a pandas DataFrame with 'temperature' and 'rolling_tempr' columns
half_t_rolling_aver = df_temperature_cleared['rolling_tempr'] / 2
t_dif_clear = df_temperature_cleared['temperature'] - df_temperature_cleared['rolling_tempr']

# Using numpy's where function for vectorized conditional assignment
df_temperature_cleared['temperature_cleared'] = np.where(abs(t_dif_clear) < half_t_rolling_aver, 
                                                         df_temperature_cleared['temperature'], 
                                                         df_temperature_cleared['rolling_tempr'])

df_temperature_cleared['event_date'] = pd.to_datetime(df_temperature_cleared['event_date'])
df_temperature_cleared.to_csv('cam_smolt_quality/data/FW_temperature_cleared.csv')

## 6. enviromental/Env_analisys.ipynb
df_env_indicators = utils.read_file('jobs environmental', READ_PARAMS, buffer=True, encoding='windows-1251') #Always from sql!!!!
df_lb_indicators = utils.read_file('logbook environmental', READ_PARAMS, buffer=True)
df_vocab = utils.read_file('dict_env', READ_PARAMS, buffer=True) #Always from drive !!!!
df_carb_dioxide_jobs = utils.read_file('jobs carbon dioxide', READ_PARAMS, buffer=True)
df_carb_dioxide_logbook = utils.read_file('logbook environmental carbon dioxide', READ_PARAMS, buffer=True)
df_transfer_1 = pd.DataFrame(columns = ['event_ts', 'locus_group_id', 'sensor_type_value', 'sensor_name', 'type_name'])
df_transfer_2 = pd.DataFrame(columns = ['event_ts', 'locus_group_id', 'sensor_type_value', 'sensor_name', 'type_name'])

n = 0 
while n < len(df_lb_indicators):
    trans_list = []
    event_ts = df_lb_indicators.iloc [n]['event_date']
    trans_list.append(event_ts)
    locus_group_id = df_lb_indicators.iloc [n]['locus_group_id']
    trans_list.append(locus_group_id)
    sensor_type_value = df_lb_indicators.iloc [n]['value']
    trans_list.append(sensor_type_value)
    logbook_event_id = df_lb_indicators.iloc [n]['logbook_event_id']
    df_vocab_filtered = df_vocab[(df_vocab['logbook_event_id'] == logbook_event_id)] 
    sensor_name = df_vocab_filtered.iloc[0]['sensor_name']
    type_name = df_vocab_filtered.iloc[0]['type_name']
    trans_list.append(sensor_name)
    trans_list.append(type_name)
    df_transfer_1.loc[len(df_transfer_1)] = trans_list
    n = n + 1

df_env_indicators = pd.concat([df_env_indicators, df_transfer_1])
df_env_indicators.reset_index(drop = True, inplace = True)

df_carb_dioxide_jobs = df_carb_dioxide_jobs.reindex(columns = df_carb_dioxide_jobs.columns.tolist()+['type_name'])
df_carb_dioxide_jobs['type_name'] = 'Carbon dioxide'
df_carb_dioxide_jobs = df_carb_dioxide_jobs.drop(columns = 'locus_id')
df_carb_dioxide_logbook = df_carb_dioxide_logbook.drop(columns = 'string')
df_carb_dioxide_logbook = df_carb_dioxide_logbook.dropna()
df_carb_dioxide_logbook.reset_index(drop = True, inplace = True)

m = 0
while m < len(df_carb_dioxide_logbook):
    trans_list_1 = []
    event_ts = df_carb_dioxide_logbook.iloc [m]['event_date']
    trans_list_1.append(event_ts)
    locus_group_id = df_carb_dioxide_logbook.iloc [m]['locus_group_id']
    trans_list_1.append(locus_group_id)
    sensor_type_value = df_carb_dioxide_logbook.iloc [m]['value']
    trans_list_1.append(sensor_type_value)
    logbook_event_id = df_carb_dioxide_logbook.iloc [m]['logbook_event_id']
    df_vocab_filtered = df_vocab[(df_vocab['logbook_event_id'] == logbook_event_id)] 
    sensor_name = df_vocab_filtered.iloc[0]['sensor_name']
    type_name = df_vocab_filtered.iloc[0]['type_name']
    trans_list_1.append(sensor_name)
    trans_list_1.append(type_name)
    df_transfer_2.loc[len(df_transfer_2)] = trans_list_1
    m = m + 1
df_carb_dioxide_jobs = pd.concat([df_carb_dioxide_jobs, df_transfer_2])
df_carb_dioxide_jobs.reset_index(drop = True, inplace = True)
df_env_indicators = pd.concat([df_env_indicators, df_carb_dioxide_jobs])
df_env_indicators.reset_index(drop = True, inplace = True)
df_env_indicators.to_csv(r'cam_smolt_quality/data/indicators_all_file_joined.csv', index = False)







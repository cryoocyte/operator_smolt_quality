import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import cam_smolt_quality.manual_run.utils as utils
import dask.dataframe as dd

# Load general data
locus_weights = utils.read_file('evt_movement_ratio_with_dates', buffer=True)
temperature = utils.read_file('FW_temperature_cleared', buffer=True)
#locus_group_matching = utils.read_file('locus_locus_group_matching', buffer=True)
fresh_water_dates = utils.read_file('FW_cycle_dates', buffer=True)
sw_fw_matching = utils.read_file('seawater_freshwater_matching', buffer=True)
mortality = utils.read_file('smolt_dataset_transfers', buffer=True)
final_locus_weighted = utils.read_file('lw_alldates_final', buffer=True)
sw_fw_matching_with_cnt = utils.read_file('sw_locus_fw_locus_population_with_counts', buffer=True)
targets = utils.read_file('targets', buffer=True)

lw_dates = locus_weights.groupby('final_locus_population_id').agg({'starttime': 'min', 'endtime': 'max'})
lw_dates['FW_cycle_length'] = (lw_dates.endtime - lw_dates.starttime).dt.days + 1  # to be checked (?)
lw_dates['starttime_year'] = lw_dates['starttime'].dt.year
lw_dates = lw_dates[lw_dates.starttime_year>=2017]

#FROM SISENSE!!!!
print('FROM SISENSE!')
vaccines =  utils.read_file('vaccines_with_final_locus_population_id', buffer=True, sep=',')
vaccines['VAC_EVENT_DATE'] = pd.to_datetime(vaccines['VAC_EVENT_DATE'] , format='%Y-%m-%d')
vaccines_agg = vaccines.groupby('FINAL_LOCUS_POPULATION_ID')[['VAC_EVENT_DATE', 'VAC_WEIGHT']].mean(numeric_only=False)
fresh_water_dates.rename(columns={'pretransfer_fw_locus_population_id': 'final_locus_population_id'}, inplace=True)
sw_fw_matching.transport_date = pd.to_datetime(sw_fw_matching.transport_date, utc=True)
sw_fw_matching_with_cnt.transfer_date = pd.to_datetime(sw_fw_matching_with_cnt['transfer_date'], utc=True)
sw_fw_matching_with_cnt['transfer_year'] = sw_fw_matching_with_cnt['transfer_date'].dt.year


## feed/feed_related_factors_upd_DE_last_8_weeks_SGR,TGC,FCR,SFR
feed_data_extended = utils.read_file('feed_data_extended_unsmoothed', buffer=True)

tempearture_mean = pd.DataFrame(temperature.groupby('final_locus_population_id')['temperature_cleared'].mean())\
.rename(columns={'temperature_cleared':'temperature_mean'})

temperature = temperature.merge(tempearture_mean,
                  left_on='final_locus_population_id', 
                  right_index=True)
feed_data_with_dates = feed_data_extended.merge(
    fresh_water_dates[['final_locus_population_id', 'first_feeding_date', 'shipout_date']],
    on='final_locus_population_id',
    how='inner',
)
feed_data_with_dates = feed_data_with_dates.merge(
    temperature[['final_locus_population_id', 'event_date', 'temperature_cleared']],
    on=['final_locus_population_id', 'event_date'],
    how='left'
)
feed_data_with_dates['temperature_cleared'].fillna(feed_data_with_dates['temperature_cleared'].mean(), inplace=True)
feed_data_with_dates = feed_data_with_dates.merge(
    vaccines_agg,
    left_on='final_locus_population_id',
    right_on='FINAL_LOCUS_POPULATION_ID',
    how='inner'
)

feed_data_with_dates = feed_data_with_dates.merge(
    sw_fw_matching_with_cnt.groupby('from_locus_population_id')[['transfer_date', 'from_avg_weight','from_count_stocking']].mean(numeric_only=False),
    left_on='final_locus_population_id',
    right_index=True,
    how='inner'
)
feed_data_with_dates['shipout_year'] = feed_data_with_dates['shipout_date'].dt.year
feed_data_with_dates['event_date'] = pd.to_datetime(feed_data_with_dates['event_date'], utc=True)
feed_data_with_dates['first_feeding_date'] = pd.to_datetime(feed_data_with_dates['first_feeding_date'], utc=True)
feed_data_with_dates['VAC_EVENT_DATE'] = pd.to_datetime(feed_data_with_dates['VAC_EVENT_DATE'], utc=True)


def get_closest_date(group):
    # Find the row in each group where 'event_date' is closest to 'first_feeding_date'
    return group.iloc[(group['event_date'] - group['VAC_EVENT_DATE']).abs().argmin()]
vaccination_feed = feed_data_with_dates.groupby('final_locus_population_id').apply(get_closest_date).reset_index(drop=True)

vaccination_feed = vaccination_feed[
    ['final_locus_population_id', 'close_count', 'VAC_WEIGHT', 'VAC_EVENT_DATE']
]
vaccination_feed.rename(columns={'VAC_WEIGHT': 'weight_vaccination'}, inplace=True)
vaccination_feed.rename(columns={'close_count': 'count_vaccination'}, inplace=True)

transfer_feed = feed_data_with_dates[
    feed_data_with_dates['event_date'] == feed_data_with_dates['transfer_date']
]
transfer_feed = transfer_feed[['final_locus_population_id', 'close_count', 'from_avg_weight', 'transfer_date']]
transfer_feed.rename(columns={'from_avg_weight': 'weight_transfer'}, inplace=True)
transfer_feed.rename(columns={'close_count': 'count_transfer'}, inplace=True)

def get_closest_date(group):
    # Find the row in each group where 'event_date' is closest to 'first_feeding_date'
    return group.iloc[(group['event_date'] - (group['transfer_date'] - pd.Timedelta(8,'w')) ).abs().argmin()]
eight_weeks_before_transfer_feed = feed_data_with_dates.groupby('final_locus_population_id').apply(get_closest_date).reset_index(drop=True)

eight_weeks_before_transfer_feed = eight_weeks_before_transfer_feed[['final_locus_population_id', 'close_count', 'close_weight_g']]#, 'transfer_date']]
eight_weeks_before_transfer_feed.rename(columns={'close_weight_g': 'weight_eight_weeks_before_transfer'}, inplace=True)
eight_weeks_before_transfer_feed.rename(columns={'close_count': 'count_eight_weeks_before_transfer'}, inplace=True)

feed_data_with_dates['event_date'] = pd.to_datetime(feed_data_with_dates['event_date'])
feed_data_with_dates['first_feeding_date'] = pd.to_datetime(feed_data_with_dates['first_feeding_date'])

def get_closest_date(group):
    # Find the row in each group where 'event_date' is closest to 'first_feeding_date'
    return group.iloc[(group['event_date'] - group['first_feeding_date']).abs().argmin()]
first_feeding_feed = feed_data_with_dates.groupby('final_locus_population_id').apply(get_closest_date).reset_index(drop=True)
first_feeding_feed = first_feeding_feed[
     ['final_locus_population_id', 'close_count', 'close_weight_g', 'first_feeding_date', ]
]
first_feeding_feed.rename(columns={'close_weight_g': 'weight_first_feeding'}, inplace=True)
first_feeding_feed.rename(columns={'close_count': 'count_first_feeding'}, inplace=True)

feed = first_feeding_feed.merge(vaccination_feed, on='final_locus_population_id')\
.merge(transfer_feed, on='final_locus_population_id').merge(eight_weeks_before_transfer_feed, on='final_locus_population_id')
feed['VAC_EVENT_DATE'] = pd.to_datetime(feed['VAC_EVENT_DATE'].dt.date, utc=True)
feed.rename(columns={'VAC_EVENT_DATE': 'vaccination_date'}, inplace=True)

feed['transfer_date'] = pd.to_datetime(feed['transfer_date'].dt.date, utc=True)
feed['first_feeding_vaccination_len'] = (feed['vaccination_date'] - feed['first_feeding_date']).dt.days
feed['first_feeding_transfer_len'] = (feed['transfer_date'] - feed['first_feeding_date']).dt.days
feed['vaccination_transfer_len'] = (feed['transfer_date'] - feed['vaccination_date']).dt.days
feed['eight_last_weeks_len'] = 56
feed_data_with_dates['event_date'] = pd.to_datetime(feed_data_with_dates['event_date'], utc=True)
feed_data_with_dates['VAC_EVENT_DATE']  = pd.to_datetime(feed_data_with_dates['VAC_EVENT_DATE'], utc=True)
feed_data_with_dates['transfer_date']  = pd.to_datetime(feed_data_with_dates['transfer_date'], utc=True)

vaccination_transfer_sum = pd.DataFrame(feed_data_with_dates[
    feed_data_with_dates['event_date'] >= feed_data_with_dates['VAC_EVENT_DATE'] 
#     feed_data_with_dates['event_date'] >= feed_data_with_dates['VAC_EVENT_DATE'] + pd.Timedelta(7,'d')
][
    feed_data_with_dates['event_date'] <= feed_data_with_dates['transfer_date']
].groupby('final_locus_population_id')['amount'].sum()).\
rename(columns={'amount': 'vaccination_transfer_sum'})

eight_last_weeks_sum = pd.DataFrame(feed_data_with_dates[
    feed_data_with_dates['event_date'] >= feed_data_with_dates['transfer_date'] - pd.Timedelta(8, 'w')
#     feed_data_with_dates['event_date'] >= feed_data_with_dates['VAC_EVENT_DATE'] + pd.Timedelta(7,'d')
][
    feed_data_with_dates['event_date'] <= feed_data_with_dates['transfer_date']
].groupby('final_locus_population_id')['amount'].sum()).\
rename(columns={'amount': 'eight_last_weeks_feed_sum'})

feed = feed.merge(vaccination_transfer_sum, how='inner', on='final_locus_population_id').\
merge(eight_last_weeks_sum, how='inner', on='final_locus_population_id')

vaccination_transfer_temperature_sum = pd.DataFrame(feed_data_with_dates[
    feed_data_with_dates['event_date'] >= feed_data_with_dates['VAC_EVENT_DATE']
][
    feed_data_with_dates['event_date'] <= feed_data_with_dates['transfer_date']
].groupby('final_locus_population_id')['temperature_cleared'].sum()).\
rename(columns={'temperature_cleared': 'vaccination_transfer_temperature_sum'})

eight_last_weeks_temperature_sum = pd.DataFrame(feed_data_with_dates[
    feed_data_with_dates['event_date'] >= feed_data_with_dates['transfer_date'] - pd.Timedelta(8, 'w')
][
    feed_data_with_dates['event_date'] <= feed_data_with_dates['transfer_date'] 
].groupby('final_locus_population_id')['temperature_cleared'].sum()).\
rename(columns={'temperature_cleared': 'eight_last_weeks_temperature_sum'})

feed = feed.merge(vaccination_transfer_temperature_sum, how='inner', on='final_locus_population_id').\
merge(eight_last_weeks_temperature_sum, how='inner', on='final_locus_population_id')


feed_calc = utils.FeedFactorsCalculator()

feed['eight_last_weeks-sgr'] = feed_calc.calculate_sgr(
    feed['weight_transfer'].values,
    feed['weight_eight_weeks_before_transfer'].values,
    feed['eight_last_weeks_len'].values
)

feed['vaccination-transfer-sgr'] = feed_calc.calculate_sgr(
    feed['weight_transfer'].values,
    feed['weight_vaccination'].values,
    feed['vaccination_transfer_len'].values
)
fcr_limit1 = -5000#(feed.count_transfer-feed.count_eight_weeks_before_transfer).quantile(.4)
fcr_limit2 = 5000#(feed.count_transfer-feed.count_eight_weeks_before_transfer).quantile(.95)
selected_flp_for_fcr=feed[(feed.count_transfer-feed.count_eight_weeks_before_transfer).between(fcr_limit1,fcr_limit2)].final_locus_population_id
feed_fcr=feed[feed.final_locus_population_id.isin(selected_flp_for_fcr)]

feed_fcr['eight_last_weeks-fcr'] = feed_calc.calculate_fcr(
    feed_fcr['eight_last_weeks_feed_sum'].values,
    feed_fcr['count_transfer'].values,
    feed_fcr['weight_transfer'].values,
    feed_fcr['count_eight_weeks_before_transfer'].values,
    feed_fcr['weight_eight_weeks_before_transfer'].values,
)

feed_fcr['vaccination-transfer-fcr'] = feed_calc.calculate_fcr(
    feed_fcr['vaccination_transfer_sum'].values,
    feed_fcr['count_transfer'].values,
    feed_fcr['weight_transfer'].values,
    feed_fcr['count_vaccination'].values,
    feed_fcr['weight_vaccination'].values,
)

feed_fcr['eight_last_weeks-fcr'] = feed_calc.calculate_fcr(
    feed_fcr['eight_last_weeks_feed_sum'].values,
    feed_fcr['count_transfer'].values,
    feed_fcr['weight_transfer'].values,
    feed_fcr['count_eight_weeks_before_transfer'].values,
    feed_fcr['weight_eight_weeks_before_transfer'].values,
)

feed_fcr['vaccination-transfer-sfr'] = feed_calc.calculate_sfr(
    feed_fcr['vaccination-transfer-fcr'],
    feed_fcr['vaccination-transfer-sgr'],
)

feed_fcr['eight_last_weeks-sfr'] = feed_calc.calculate_sfr(
    feed_fcr['eight_last_weeks-fcr'],
    feed_fcr['eight_last_weeks-sgr'],
)
feed['vaccination-transfer-tgc'] = 1000 * (
    (feed['weight_transfer']**(1/3)-feed['weight_vaccination']**(1/3)) / feed['vaccination_transfer_temperature_sum']
)

feed['eight_last_weeks-tgc'] = 1000 * (
    (feed['weight_transfer']**(1/3)-feed['weight_eight_weeks_before_transfer']**(1/3)) / feed['eight_last_weeks_temperature_sum']
)

mortality_cols = ['locus_id','fish_group_id','transfer_year','transfer_month', 'transfer_month_year',
                  'transfer_season', 'transfer_season2','total_count','total_mortality_perc_90',
                  'to_avg_weight','transfer_season2']

sw_cols = ['to_locus_id', 'to_fish_group_id',
           'transfer_date', 'from_locus_population_id',
           'from_count_stocking','transfer_year']

mortality_final_locus = mortality[mortality_cols].merge(
    sw_fw_matching_with_cnt[sw_cols],
    left_on=['fish_group_id', 'locus_id','transfer_year'],
    right_on=['to_fish_group_id', 'to_locus_id','transfer_year'],
    how='left'
)

mortality_final_locus['from_locus_population_id'] = mortality_final_locus['from_locus_population_id'].fillna(0).astype('int32')
feed_factors_locus = feed.merge(
    mortality_final_locus[['from_locus_population_id', 'locus_id','fish_group_id', 'transfer_year', 'from_count_stocking']],
    left_on=['final_locus_population_id',],
    right_on=['from_locus_population_id',],
    how='inner')

factors = [
#     'first_feeding-transfer-sgr',
#     'first_feeding-vaccination-sgr',
    'vaccination-transfer-sgr',
#     'first_feeding-transfer-fcr',
#     'first_feeding-vaccination-fcr',
#     'vaccination-transfer-fcr',
#     'first_feeding-transfer-sfr',
#     'first_feeding-vaccination-sfr',
#     'vaccination-transfer-sfr',
    'vaccination-transfer-tgc',
    'eight_last_weeks-sgr',
#     'eight_last_weeks-fcr',
#     'eight_last_weeks-sfr',
    'eight_last_weeks-tgc',
]

key_columns = ['locus_id','fish_group_id','transfer_year']


feed_factors = utils.create_factors_df(feed_factors_locus,
                                 factors,
                                 key_columns, 
                                 weight_column='from_count_stocking',
                                 weighted_func=utils.weighted_avg)

feed_factors = feed_factors.merge(
    pd.DataFrame(mortality_final_locus.groupby(key_columns).agg({'total_mortality_perc_90':np.mean,
                                                                'to_avg_weight':np.mean
#                                                                 'transfer_season2':np.min
                                                                })),
    on=key_columns,
    how='inner'
)

feed_factors.rename(columns={'total_mortality_perc_90': 'mortality'}, inplace=True)
feed_factors.to_csv('cam_smolt_quality/data/factors_feed_period_DE_SGR_TGC.csv', index=False)

significant_factors = ['locus_id', 'fish_group_id', 'transfer_year','eight_last_weeks-sgr','eight_last_weeks-tgc']
feed_factors_combined_shortlisted=feed_factors[significant_factors]
feed_factors_combined_shortlisted.to_csv('cam_smolt_quality/data\\significant_factors_feed_part2.csv',index=False)

##### FCR + SFR TO BE ADDED #####

## feed/feed_producer_processing.ipynb
feed_data = utils.read_file('feed_UPS_Petrohue', buffer=True)
feed_data.rename(columns={'start_reg_time': 'event_date'}, inplace=True)
feed_data['event_date'] = pd.to_datetime(feed_data['event_date'], utc=True) #format=TIME_FORMAT
feed_data['event_year'] = feed_data['event_date'].dt.year

feed_catalog = utils.read_file('dict_feed_name', buffer=True)
feed_catalog['producer'] = feed_catalog['feed_name'].apply(lambda x: x.split()[0])
feed_data = feed_data.merge(feed_catalog[['mrts_feed_id', 'producer']], how='inner', on='mrts_feed_id')

feed_data_final_lp = feed_data.merge(final_locus_weighted,
                                     how='inner',
                                     left_on=['event_date', 'locus_id'],
                                     right_on=['event_date', 'historic_locus_id'])
feed_data_final_lp = feed_data_final_lp.merge(vaccines_agg,
                                              how='left',
                                              left_on='final_locus_population_id',
                                              right_index=True)
final_locus_population_transfer_dates = sw_fw_matching_with_cnt[['from_locus_population_id','transfer_date']].groupby('from_locus_population_id').max()
feed_data_final_lp = feed_data_final_lp.merge(final_locus_population_transfer_dates,
                                     how='left',
                                     left_on='final_locus_population_id',
#                                      right_on='from_locus_population_id'
                                     right_index=True
                                             )
feed_data_final_lp['eight_weeks_before_transfer'] = feed_data_final_lp['transfer_date'] - pd.Timedelta(8,'w')
encoded_producer = pd.get_dummies(feed_data_final_lp['producer'])
feed_data_final_lp = feed_data_final_lp.drop('producer', axis=1)
feed_data_final_lp = feed_data_final_lp.join(encoded_producer)
feed_data_final_lp.to_csv('cam_smolt_quality/data\\feed_data_final_lp.csv',index=False)

feed_data_final_lp['event_year'] = feed_data_final_lp['event_date'].dt.year
for col in tqdm(encoded_producer.columns):
    feed_data_final_lp[f'weight_{col}'] = feed_data_final_lp['weight'] * feed_data_final_lp[col]
weighted_producer_cols = ['weight_' + col for col in encoded_producer.columns]
feed_data_final_lp_grouped = feed_data_final_lp.\
groupby(['final_locus_population_id','event_date'])[weighted_producer_cols].sum()
feed_data_final_lp_eight_last_weeks=feed_data_final_lp[feed_data_final_lp.event_date.between(feed_data_final_lp['eight_weeks_before_transfer'],feed_data_final_lp['transfer_date'])]
feed_data_final_lp_vacc_transfer=feed_data_final_lp[feed_data_final_lp.event_date.between(feed_data_final_lp['VAC_EVENT_DATE'],feed_data_final_lp['transfer_date'])]
feed_data_final_lp_grouped = feed_data_final_lp_grouped.reset_index()
feed_data_final_lp_grouped['weight_sum'] = feed_data_final_lp_grouped[weighted_producer_cols].sum(axis=1)
for col in weighted_producer_cols:
    feed_data_final_lp_grouped[col] = feed_data_final_lp_grouped[col] / feed_data_final_lp_grouped['weight_sum']
#if at least 50% of tanks were without feeding info
feed_data_final_lp_grouped['low_feeding_flag']=np.zeros(len(feed_data_final_lp_grouped))
feed_data_final_lp_grouped.loc[feed_data_final_lp_grouped['weight_sum']<0.5,'low_feeding_flag']=1
#if have two records on feed with different mrts_feed_id in the same date
feed_data_final_lp_grouped['double_feeding_flag']=np.zeros(len(feed_data_final_lp_grouped))
feed_data_final_lp_grouped.loc[feed_data_final_lp_grouped['weight_sum']>1.5,'double_feeding_flag']=1

feed_data_final_lp_grouped['low_feeding_flag']=feed_data_final_lp_grouped.low_feeding_flag.astype('int')
feed_data_final_lp_grouped['double_feeding_flag']=feed_data_final_lp_grouped.double_feeding_flag.astype('int')
feed_producer_final_lp = feed_data_final_lp_grouped.groupby('final_locus_population_id').mean()
feed_producer_final_lp.to_csv('../data\\feed_producer_final_lp_ALL_DATES.csv')

#WOORK HERE

key_columns = ['locus_id','fish_group_id','transfer_year']
mortality_cols = ['locus_id', 'fish_group_id','transfer_year', 'transfer_month',
                  'transfer_month_year', 'transfer_season', 'transfer_season2', 'total_count', 'total_mortality_perc_90']
sw_cols = ['to_locus_id', 'to_fish_group_id', 'transfer_date',
           'from_locus_population_id', 'from_count_stocking', 'transfer_year']
mortality_final_locus = mortality[mortality_cols].merge(
    sw_fw_matching_with_cnt[sw_cols],
    left_on=['fish_group_id', 'locus_id','transfer_year'],
    right_on=['to_fish_group_id', 'to_locus_id','transfer_year'],
    how='left'
)
mortality_final_locus['from_locus_population_id'] = mortality_final_locus['from_locus_population_id'].fillna(0).astype('int32')

feed_producer_final_lp = feed_producer_final_lp.merge(
    mortality_final_locus[['from_locus_population_id', 'locus_id','fish_group_id', 'transfer_year', 'from_count_stocking']],
    left_index=True,
    right_on=['from_locus_population_id',],
    how='right')

factors = feed_producer_final_lp.columns.difference(
    ['from_locus_population_id', 'locus_id', 'fish_group_id', 'transfer_year', 'from_count_stocking']
)

feed_producer = utils.create_factors_df(feed_producer_final_lp,
                                     factors,
                                     key_columns,
                                     weight_column='from_count_stocking',
                                     weighted_func=utils.weighted_avg)
feed_producer = feed_producer.merge(
    pd.DataFrame(mortality_final_locus.groupby(key_columns)['total_mortality_perc_90'].mean()),
    on=key_columns,
    how='inner')

feed_producer.rename(columns={'total_mortality_perc_90': 'mortality'}, inplace=True)
feed_producer.to_csv(f'cam_smolt_quality/data/factors_{main_factor.lower()}.csv', index=False)


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
locus_group_matching = utils.read_file('locus_locus_group_matching', buffer=True)
fresh_water_dates = utils.read_file('FW_cycle_dates', buffer=True)
sw_fw_matching = utils.read_file('seawater_freshwater_matching', buffer=True)
mortality = utils.read_file('smolt_dataset_transfers', buffer=True)
final_locus_weighted = utils.read_file('lw_alldates_final', buffer=True)
sw_fw_matching_with_cnt = utils.read_file('sw_locus_fw_locus_population_with_counts', buffer=True)

lw_dates = locus_weights.groupby('final_locus_population_id').agg({'starttime': 'min', 'endtime': 'max'})
lw_dates['FW_cycle_length'] = (lw_dates.endtime - lw_dates.starttime).dt.days + 1  # to be checked (?)
lw_dates['starttime_year'] = lw_dates['starttime'].dt.year
lw_dates = lw_dates[lw_dates.starttime_year>=2017]


## 7. mortality/preprocess_reason.ipynb
fw_mortality = utils.read_file('fw_mortality', buffer=True)
inventory = utils.read_file('inventory_Petrohue_UPS', buffer=True)
mortality_ref = utils.read_file('dict_mortality', encoding="ISO-8859-1", buffer=True) #Always true

fw_mortality = fw_mortality.merge(mortality_ref,left_on='mortality_reason_id', right_on='id', how='left')
reason_exclude = 'Eliminación Productiva'
fw_mortality = fw_mortality[fw_mortality['mortality_reason'] != reason_exclude]
fw_mortality = fw_mortality.merge(inventory, on=['event_date', 'locus_id'], how='left')
fw_mortality['event_date'] = pd.to_datetime(fw_mortality['event_date'])
fw_mortality['open_count'] = fw_mortality.groupby('locus_id')['open_count'].apply(lambda group: group.interpolate(method='linear')).reset_index(level=0, drop=True)
fw_mortality.drop(fw_mortality[fw_mortality['open_count'] == 0][fw_mortality['close_count'] == 0].index, inplace=True)
fw_mortality['open_count'][fw_mortality['open_count'] == 0] = fw_mortality['close_count']  # + fw_mortality['mortality_coun']
fw_mortality['mortality_rate'] = fw_mortality['mortality_count'] / fw_mortality['open_count']
fw_mortality.to_csv('cam_smolt_quality/data/fw_mortality.csv', index=False)

mortality_reasons = fw_mortality.groupby(['mortality_reason'])['mortality_count'].sum().sort_values(ascending=False)[:10].index.tolist() # ['Desadaptado', ]
fw_mortality_by_reason = {}
for reason in mortality_reasons:
    fw_mortality_by_reason[reason] = fw_mortality[fw_mortality['mortality_reason'] == reason]
for reason in tqdm(fw_mortality_by_reason):
    fw_mortality_by_reason[reason] = fw_mortality_by_reason[reason].merge(
        final_locus_weighted,
        how='inner', 
        left_on=['event_date', 'locus_id'],
        right_on=['event_date', 'historic_locus_id']
    )
    fw_mortality_by_reason[reason]['event_year'] = fw_mortality_by_reason[reason]['event_date'].dt.year
    fw_mortality_by_reason[reason]['weighted_mortality_rate'] = fw_mortality_by_reason[reason]['weight'] * fw_mortality_by_reason[reason]['mortality_rate']

    fw_mortality_by_reason[reason] = fw_mortality_by_reason[reason].\
    groupby(['final_locus_population_id','event_date'])[['weighted_mortality_rate']]\
    .agg(lambda x: x.sum(skipna=False)).reset_index()
    fw_mortality_by_reason[reason].rename(columns={'weighted_mortality_rate': 'mortality_rate'},inplace=True)

fw_mortality_final_locus_grouped = pd.concat([fw_mortality_by_reason[rsn] for rsn in mortality_reasons],axis=0).reset_index(drop=True)
fw_mortality_final_locus_grouped.to_csv('cam_smolt_quality/data/fw_mortality_final_lp_grouped_no_productiva.csv', index=False)

for reason in tqdm(fw_mortality_by_reason):
    fw_mortality_by_reason[reason] = fw_mortality_by_reason[reason].merge(
        fresh_water_dates,
        left_on='final_locus_population_id',
        right_on='pretransfer_fw_locus_population_id',
        how='inner'
    )
 
fw_mortality_by_reason_full_range = {}
for reason in tqdm(fw_mortality_by_reason):
    reason_df = fw_mortality_by_reason[reason]
    by_id = []
    for _id in reason_df['final_locus_population_id'].unique():
        reason_df_by_id = reason_df[reason_df['final_locus_population_id'] == _id]
        full_range = pd.DataFrame(
            pd.date_range(
                start=reason_df_by_id['first_movement_date'].iloc[0],
                end=reason_df_by_id['shipout_date'].iloc[0]),
            columns=['event_date']
        )
        mortality_full_range = full_range.merge(reason_df_by_id, how='left', on='event_date')
        mortality_full_range['mortality_rate'].fillna(0, inplace=True)
        mortality_full_range['final_locus_population_id'].fillna(_id, inplace=True)
        mortality_full_range['final_locus_population_id'] = mortality_full_range['final_locus_population_id']\
        .astype(int)
        
        by_id.append(mortality_full_range)

    fw_mortality_by_reason_full_range[reason] = pd.concat(by_id)
for reason in fw_mortality_by_reason_full_range:
    fw_mortality_by_reason_full_range[reason][
        ['event_date', 'final_locus_population_id', 'mortality_rate']
    ].to_csv(f'cam_smolt_quality/data/mrts/fw_mortality_{reason}.csv', index=False)
del(fw_mortality,)

## 8.treatment/treatment_analysis_DE_upd_work.ipynb
treatment = utils.read_file('evt_treatment_UPS_Petrohue', buffer=True)
substance = utils.read_file('lkp_active_substance', buffer=True, encoding='windows-1252', sep='\t') #Always True
treatment_lkp = utils.read_file('lkp_treatment_method', buffer=True, encoding='windows-1252', sep='\t') #Always True
tarp_lkp = utils.read_file('lkp_tarp_type', buffer=True, encoding='windows-1252', sep='\t') #Always True

substance.rename(columns={'id': 'active_substance_id'}, inplace=True)
substance.rename(columns={'name': 'active_substance_name'}, inplace=True)
treatment_lkp.rename(columns={'id': 'treatment_method_id'}, inplace=True)
treatment_lkp.rename(columns={'name': 'treatment_method_name'}, inplace=True)
tarp_lkp.rename(columns={'id': 'tarp_type_id'}, inplace=True)
tarp_lkp.rename(columns={'name': 'tarp_type_name'}, inplace=True)
treatment = treatment.merge(substance[['active_substance_id', 'active_substance_name']], how='left', on='active_substance_id')
treatment = treatment.merge(treatment_lkp[['treatment_method_id', 'treatment_method_name']], how='left', on='treatment_method_id')
treatment = treatment.merge(tarp_lkp[['tarp_type_id', 'tarp_type_name']], how='left', on='tarp_type_id')
treatment['event_date'] = pd.to_datetime(treatment['event_date'])
treatment['event_year'] = treatment['event_date'].dt.year
treatment_diff = treatment.groupby(['locus_id', 'locus_population_id', 'prescription', 'treatment_method_id'])['event_date'].agg(['max','min']).reset_index()
treatment_diff['duration'] = treatment_diff['max'] - treatment_diff['min']

na_list = [x for x in treatment.prescription.dropna().unique() if len(x)<5]
treatment['prescription'] = treatment['prescription'].replace(na_list, np.nan)
treatment['prescription'] = treatment['prescription'].str.upper().replace(' M0002APHU', 'LM0002APHU').fillna(0)

fungosis_list = ['fungus','fumgosis','funosis','fungodid']
flavobacteria_list = ['flavobacteriosis','flavobacterosis','flabobacteriosis','bacteria','flabovacteriosis']
treatment['reason'] = treatment['reason'].str.lower()
treatment['reason'] = treatment['reason'].replace(fungosis_list, 'fungosis').replace(flavobacteria_list, 'flavobacteria').fillna(0)

treatment_grouped = treatment.groupby(
    ['prescription', 'reason', 'treatment_method_id', 'locus_population_id', 'locus_id']
).agg({'amount': 'sum',
       'active_substance_name': lambda col: col.mode().iloc[0] if len(col.mode()) > 0 else np.nan,
      'treatment_method_name': lambda col: col.mode().iloc[0] if len(col.mode()) > 0 else np.nan,
      'tarp_type_name': lambda col: col.mode().iloc[0] if len(col.mode()) > 0 else np.nan,
      'event_date': ['min', 'max']
      }).reset_index()

treatment_grouped.columns = ['prescription', 'reason', 'treatment_method_id', 'locus_population_id', 'locus_id',
                             'amount', 'active_substance_name', 'treatment_method_name', 'tarp_type_name',
                             'min_event_date', 'max_event_date']
treatment_grouped['duration'] = (treatment_grouped['max_event_date'] - treatment_grouped['min_event_date']).dt.total_seconds() / (60 * 60 * 24)
treatment_grouped['min_event_date_year'] = treatment_grouped['min_event_date'].dt.year
treatment_methods = treatment_grouped['treatment_method_name'].unique()
treatment_grouped = treatment_grouped.join(pd.get_dummies(treatment_grouped['treatment_method_name']))
for method in treatment_methods:
    treatment_grouped[method] = treatment_grouped[method] * treatment_grouped['duration']
    treatment_grouped.rename(columns={method: f'{method}_duration'}, inplace=True)
active_substances = treatment_grouped['active_substance_name'].dropna().unique()
treatment_grouped = treatment_grouped.join(pd.get_dummies(treatment_grouped['active_substance_name']))
for substance in active_substances:
    treatment_grouped[substance] = treatment_grouped[substance] * treatment_grouped['duration']
    treatment_grouped.rename(columns={substance: f'{substance}_duration'}, inplace=True)
tarp_names = treatment_grouped['tarp_type_name'].dropna().unique()
treatment_grouped = treatment_grouped.join(pd.get_dummies(treatment_grouped['tarp_type_name']))
for tarp_name in tarp_names:
    treatment_grouped[tarp_name] = treatment_grouped[tarp_name] * treatment_grouped['duration']
    treatment_grouped.rename(columns={tarp_name: f'{tarp_name}_duration'}, inplace=True)
treatment_grouped = treatment_grouped.join(pd.get_dummies(treatment['active_substance_name']))
for substance in active_substances:
    treatment_grouped[substance] = treatment_grouped[substance] * treatment_grouped['amount']
    treatment_grouped.rename(columns={substance: f'{substance}_amount'}, inplace=True)
treatment_grouped_full = treatment_grouped.copy()
treatment_grouped = treatment_grouped[treatment_grouped.min_event_date_year>=2015]
treatment = treatment_grouped.copy()
del(treatment_grouped)
treatment['min_event_date'] = pd.to_datetime(treatment['min_event_date'], utc=True) #, dayfirst=True
treatment['max_event_date'] = pd.to_datetime(treatment['max_event_date'], utc=True) #, dayfirst=True
treatment['min_event_date'] = treatment['min_event_date'].dt.date
treatment['max_event_date'] = treatment['max_event_date'].dt.date

def dummify(df, feature: str):
    dummy = pd.get_dummies(df[feature])
    for col in dummy.columns:
        dummy.rename(columns={col: f'is_{col}'}, inplace=True)
    df = df.join(dummy)
    return df

treatment['reason'] = treatment['reason'].replace('fungosis', 'micosis')

treatment = dummify(treatment, 'active_substance_name')
treatment = dummify(treatment, 'treatment_method_name')
treatment = dummify(treatment, 'reason')

treatment.rename(columns={'is_0': 'is_no_reason'}, inplace=True)
treatment['is_prescription'] = np.where(treatment['prescription'] != '0', 1, 0)
def range_date(group):
    is_cols = [col for col in group.columns if 'is_' in col]    
    grouped = []

    for idx, row in group.iterrows():
        data_range = pd.date_range(row['min_event_date'], row['max_event_date'])
        is_data = group[is_cols]
        data = pd.DataFrame({'event_date': data_range})
        data['is_treated'] = 1
        for col in is_cols:            
            data[col] = row[col]
        grouped.append(data)

    return pd.concat(grouped)
treatment_ts = treatment.groupby('locus_id').apply(range_date).reset_index(level=0)
treatment_ts = treatment_ts.reset_index(drop=True)

#Dmitrii`s update
def range_date_full(group):
    is_cols = [col for col in group.columns if 'is_' in col]
    full_range_df = pd.DataFrame({'event_date': pd.date_range(group['event_date'].min(), group['event_date'].max())})
    full_range_df['locus_id'] = group['locus_id'].iloc[0]
    merged_group = pd.merge(full_range_df, group, on=['event_date', 'locus_id'], how='left')
    for col in is_cols:
        merged_group[col] = merged_group[col].fillna(0)
    merged_group = merged_group.sort_values('event_date')
    return merged_group.drop('locus_id', axis=1)
treatment_ts_full_range = treatment_ts.groupby('locus_id').apply(range_date_full).reset_index(drop=False)
treatment_ts_full_range.to_csv('cam_smolt_quality/data/treatment_ts_full_range_raw.csv', index=False)

treatment_ts_full_range['event_date'] = pd.to_datetime(treatment_ts_full_range['event_date'], utc=True)
treatment_ts_full_range = treatment_ts_full_range.groupby(['locus_id', 'event_date']).sum().reset_index()

treatment_ts_full_range[[col for col in treatment_ts_full_range.columns if 'is_' in col]] = np.where(
    treatment_ts_full_range[[col for col in treatment_ts_full_range.columns if 'is_' in col]] > 1,
    1,
    treatment_ts_full_range[[col for col in treatment_ts_full_range.columns if 'is_' in col]])
treatment_ts_full_range.to_csv('cam_smolt_quality/data/treatment_ts_full_range.csv', index=True)


list_substances=['is_BRONOPOL', 'is_CLORAMINA T', 'is_Lufenurón', 'is_NACL', 'is_SAPROSAFE AW']
list_reasons = ['is_no_reason', 'is_caligus ', 'is_flavobacteria', 'is_ipn', 'is_micosis', 'is_otro']

tmp_list = ['is_treated','is_BRONOPOL','is_CLORAMINA T','is_Lufenurón','is_NACL','is_SAPROSAFE AW','is_Bath','is_Baño sin PMV','is_Feed']
short_list = [x for x in treatment_ts_full_range.columns if 'is_' in x]
treatment_ts_full_range_flp = treatment_ts_full_range[short_list + ['event_date', 'locus_id']].merge(final_locus_weighted,
                   how='inner', 
                   left_on=['event_date', 'locus_id'],
                   right_on=['event_date', 'historic_locus_id'])
is_cols = [col for col in treatment_ts_full_range_flp.columns if 'is_' in col]    
for col in is_cols:
    try:
        treatment_ts_full_range_flp[f'{col}_weighted'] = \
        treatment_ts_full_range_flp['weight'] * treatment_ts_full_range_flp[col]
    except TypeError:
        treatment_ts_full_range_flp[f'{col}_weighted'] = \
        treatment_ts_full_range_flp['weight'] * treatment_ts_full_range_flp[col].astype(bool)
#Optimized!
sum_columns = [f'{col}_weighted' for col in is_cols]
# Fill NaN values with 0 before the groupby operation
treatment_ts_full_range_flp = treatment_ts_full_range_flp.fillna(0)
# Perform the groupby and sum operation
treatment_ts_full_range_flp_grouped = treatment_ts_full_range_flp.groupby(['final_locus_population_id', 'event_date'])[sum_columns].sum().reset_index()
treatment_ts_full_range_flp_grouped.to_csv('cam_smolt_quality/data/treatment_ts_full_range_flp_grouped_DE.csv', index=False)
del(treatment_ts_full_range_flp_grouped, treatment_ts_full_range, treatment_ts_full_range_flp)

# 10.feed/preprocess_feed_DE.ipynb
feed_data = utils.read_file('feed_UPS_Petrohue', buffer=True)
feed_data = feed_data.merge(locus_group_matching, how='inner', on='locus_id')
feed_data.rename(columns={'start_reg_time': 'event_date'}, inplace=True)
feed_data['event_date'] = pd.to_datetime(feed_data['event_date']) #, format=TIME_FORMAT
feed_data['event_year'] = feed_data['event_date'].dt.year
feed_data = feed_data.drop_duplicates()

locus_avg_weight = utils.read_file('FW_locus_weight_UPS_Petrohue', buffer=True)
locus_avg_weight['open_biomass_kg'] = locus_avg_weight ['open_count'] * locus_avg_weight ['open_weight_g'] / 1000
locus_avg_weight['close_biomass_kg'] = locus_avg_weight ['close_count'] * locus_avg_weight ['close_weight_g'] / 1000
locus_avg_weight['event_date'] = pd.to_datetime(locus_avg_weight['event_date'], utc=True)
locus_avg_weight=locus_avg_weight[locus_avg_weight.close_count>0]

#grouping, merging feed amount data and locus average weight data to calculate SFR, filtering
feed_data_by_locus = pd.DataFrame(feed_data.groupby(['locus_id', 'event_date'])['amount'].mean()).reset_index()
feed_data_by_locus=feed_data_by_locus.merge(locus_avg_weight,how='left')
feed_data_by_locus=feed_data_by_locus[feed_data_by_locus.amount<feed_data_by_locus.amount.quantile(.999)]
feed_data_by_locus=feed_data_by_locus[feed_data_by_locus.close_biomass_kg!=feed_data_by_locus.open_biomass_kg]
feed_data_by_locus=feed_data_by_locus[feed_data_by_locus.open_biomass_kg>0]
feed_data_by_locus['SFR'] = feed_data_by_locus['amount'] / feed_data_by_locus['open_biomass_kg'] 
feed_data_by_locus['SGR'] = 100 * ( feed_data_by_locus['close_weight_g'] / feed_data_by_locus['open_weight_g'] - 1)
feed_data_by_locus['FCR'] = feed_data_by_locus['amount'] / (feed_data_by_locus['close_biomass_kg']-feed_data_by_locus['open_biomass_kg'])

q1 = feed_data_by_locus['SGR'].quantile(.99)
q2 = feed_data_by_locus['FCR'].quantile(.99)
feed_data_by_locus = feed_data_by_locus[feed_data_by_locus.SGR.between(0,q1)]
print(len(feed_data_by_locus))
feed_data_by_locus = feed_data_by_locus[feed_data_by_locus.FCR.between(0,q2)]

df_dates_locus = fresh_water_dates.merge(
    lw_dates.reset_index()[['final_locus_population_id']],
    left_on='pretransfer_fw_locus_population_id',
    right_on='final_locus_population_id',
    how='inner'
)
df_dates_locus.drop(columns=['final_locus_population_id'], inplace=True)

tmp_list = []
for idx, row in df_dates_locus.iterrows():
    lp = row.pretransfer_fw_locus_population_id
    start = row.first_movement_date
    end = row.shipout_date
    for d in pd.date_range(start, end):
        tmp_list.append([lp, d])
tmp_df = pd.DataFrame(tmp_list, columns=['final_locus_population_id','event_date'])

factor_list = [ 'amount', 'SFR','FCR','SGR', 'open_count', 'open_weight_g', 'close_count', 'close_weight_g']

feed_data_final_lp_grouped = []
for locus_id, fdbl in tqdm(feed_data_by_locus.groupby('locus_id')): 

    flw = final_locus_weighted[final_locus_weighted['historic_locus_id'] == locus_id]
    lgm = locus_group_matching[locus_group_matching['locus_id'] == locus_id]
    fllgw = flw.merge(
        lgm,
        left_on='historic_locus_id',
        right_on='locus_id',
        how='left'
    )
    fllgw.drop(columns='locus_id',inplace=True)
    
    fdfl = fllgw[['final_locus_population_id', 'event_date', 'historic_locus_id','weight']].merge(
        fdbl, how='inner', left_on=['historic_locus_id','event_date'], right_on=['locus_id','event_date'])[['final_locus_population_id', 'event_date',
           'weight', 'locus_id','amount','SFR','FCR','SGR','open_count', 'open_weight_g','close_count', 'close_weight_g']]
                                                                                                            
    fdfl['event_year'] = fdfl['event_date'].dt.year
    for factor in factor_list:
        fdfl[f'weight_{factor}'] = fdfl['weight']*fdfl[factor]                                                                                      
              
    # ffdfl_grouped=fdfl.groupby(['final_locus_population_id','event_date'])[[f'weight_{x}' for x in factor_list]+['weight']].agg(lambda x: x.sum(skipna=False)).reset_index()
    
    columns_to_sum = [f'weight_{x}' for x in factor_list] + ['weight']
    ffdfl_grouped = fdfl.groupby(['final_locus_population_id', 'event_date'])[columns_to_sum].sum().reset_index()
                   
    for factor in factor_list:
        ffdfl_grouped[f'weight_{factor}'] = ffdfl_grouped[f'weight_{factor}'] / ffdfl_grouped['weight']
        ffdfl_grouped.rename(columns={f'weight_{factor}':f'{factor}'},inplace=True)
    ffdfl_grouped.drop(columns=['weight'],inplace=True)
                                                     
    feed_data_final_lp_grouped.append(ffdfl_grouped)    
feed_data_final_lp_grouped = pd.concat(feed_data_final_lp_grouped, axis=0).reset_index(drop=True)
feed_data_final_lp_grouped = feed_data_final_lp_grouped.drop_duplicates()



feed = tmp_df.merge(feed_data_final_lp_grouped, how='left')
del(feed_data_final_lp_grouped)
feed_interp = pd.DataFrame()
for idx, df in tqdm(feed.groupby('final_locus_population_id')):
    tmp_df2 = df.copy()
    for factor in factor_list:
        tmp_df2[factor] = df[factor].interpolate()
    feed_interp = pd.concat([feed_interp, tmp_df2])
feed_interp.dropna().to_csv('cam_smolt_quality/data/feed_data_extended_unsmoothed.csv',index=False)



feed_data_smoothed = []
for full_cycle_id in tqdm(feed_interp.final_locus_population_id.unique()):
    ph_full_cycle = feed_interp[feed_interp['final_locus_population_id'] == full_cycle_id]
    smoothed_ph_full_cycle = ph_full_cycle.copy()
    smoothed_ph_full_cycle['SFR'] = smoothed_ph_full_cycle['SFR'].rolling(7, min_periods=1).mean()

    feed_data_smoothed.append(smoothed_ph_full_cycle)
feed_data_smoothed = pd.concat(feed_data_smoothed)

feed_interp.to_csv('cam_smolt_quality/data/feed_SFR_data_unsmoothed.csv', index=False)
feed_data_smoothed.to_csv('cam_smolt_quality/data/feed_SFR_data_smoothed.csv', index=False)


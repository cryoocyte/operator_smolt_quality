import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils

from cam_smolt_quality.configs.cam_config import config


req_sites = pd.DataFrame([
        ['Pilpilehue', '2022-07-01', '2022-08-31'],
        ['Loncochalgua', '2022-09-01','2022-09-30'],
        ['Marilmo', '2022-10-01','2022-10-31'],
        ['Ahoni', '2022-11-01','2022-11-30'],
        ['Johnson 2', '2023-01-01','2023-01-31'],
    ], 
    columns=['site_name', 'from_date', 'to_date']
)
req_sites['from_date'] = pd.to_datetime(req_sites['from_date'], utc=True)
req_sites['to_date'] = pd.to_datetime(req_sites['to_date'], utc=True)

stock_df = extr.extract_data('stockings_mrts', config)
stock_df = stock_df[stock_df['to_site_name'].isin(req_sites['site_name'].tolist())]

g_dfs = []
for g_id, g_df in stock_df.groupby('to_site_name'):
    r_df = req_sites[req_sites['site_name']==g_id]
    g_df = g_df[((g_df['transfer_date']>=r_df['from_date'].item()) & (g_df['transfer_date']<=r_df['to_date'].item()))]
    g_dfs.append(g_df)
stock_df = pd.concat(g_dfs, axis=0).reset_index(drop=True)
# stock_df.to_csv('data/demand/cam_stockings_20240115.csv')

add_args = {'DST_LPS': tuple(stock_df['from_lp_id'].unique().tolist())}
movements_ratio_df = extr.extract_data('movements_ratio_mrts', config, additional_args=add_args)
final_lps = dict(zip(movements_ratio_df['historic_lp_id'], movements_ratio_df['final_lp_id']))

add_args = {'LP_IDS': tuple(movements_ratio_df['historic_lp_id'].unique().tolist())}


keys = ['event_date', 'site_name', 'locus_id', 'locus_name', 'lp_id', 'fish_group']

atpasa_df = extr.extract_data('lab_atpasa_mrts', config, additional_args=add_args)
atpasa_df = utils.add_prefix(atpasa_df, keys=keys, prefix_name='atpasa')
# atpasa_df['final_lp_id'] = atpasa_df['lp_id'].replace(final_lps)
# atpasa_df = pd.merge(atpasa_df, stock_df[['from_lp_id', 'to_site_name']].rename({'from_lp_id': 'final_lp_id', 'to_site_name': 'final_site_name'}, axis=1), on='final_lp_id', how='left')
# atpasa_df.to_csv('cam_smolt_quality/data/demand/cam_atpasa_20240115.csv', index=False)


ltlg_df = extr.extract_data('locus_to_locus_group_mrts', config)
inv_df = extr.extract_data('inventory_mrts', config, additional_args=add_args)

inv_df['final_lp_id'] = inv_df['lp_id'].replace(final_lps)
ltlg_dict = dict(zip(ltlg_df['locus_id'], ltlg_df['locus_group_id']))
inv_df['locus_group_id'] = inv_df['locus_id'].replace(ltlg_dict)
inv_df = pd.merge(inv_df, stock_df[['from_lp_id', 'to_site_name']].rename({'from_lp_id': 'final_lp_id', 'to_site_name': 'final_site_name'}, axis=1), on='final_lp_id', how='left')

mortality_df = extr.extract_data('mortality_mrts', config, additional_args=add_args)
rsns = mortality_df.groupby(['mortality_reason'])['mortality_count'].sum().sort_values(ascending=False).index[:10].tolist()
mortality_df = mortality_df[mortality_df['mortality_reason'].isin(rsns)].pivot_table(index=keys, columns='mortality_reason', values='mortality_count', aggfunc='sum').reset_index()
mortality_df = utils.add_prefix(mortality_df, keys=keys, prefix_name='mortality')

treatment_df = extr.extract_data('treatments_mrts', config, additional_args=add_args)
treatment_df = treatment_df.pivot_table(index=keys, columns='active_substance_name', values='amount', aggfunc='sum').reset_index()
treatment_df = utils.add_prefix(treatment_df, keys=keys, prefix_name='active_substance_name')

feed_df = extr.extract_data('feed_mrts', config, additional_args=add_args)
feed_df = feed_df.groupby(keys)['feed_amount'].sum().reset_index()
feed_df = utils.add_prefix(feed_df, keys=keys, prefix_name='feed')

jobs_df = extr.extract_data('jobs_test_mrts', config, skip_full_format=True)
logbook_df = extr.extract_data('logbook_mrts', config)

locus_df = inv_df[['locus_id', 'locus_name']].drop_duplicates()
sensors_df = pd.concat([logbook_df, jobs_df], axis=0)
sensors_df = sensors_df.drop_duplicates(keep='last') #Keep jobs data
sensors_df = sensors_df[sensors_df['variable'].notna()]
# del(jobs_df, logbook_df)
sensors_df = sensors_df.pivot_table(values='value', columns=['variable'], index=['event_date', 'locus_group_id'], aggfunc='mean')
sensors_df = utils.add_prefix(sensors_df, keys='', prefix_name='sensor').reset_index()

light_df = extr.extract_data('light_regime_mrts', config, additional_args=add_args)
light_df = utils.add_prefix(light_df, keys=keys, prefix_name='light_regime')

inv_df = reduce(
    lambda left, right: pd.merge(left, right, on=keys, how='left'),
    [inv_df, mortality_df, treatment_df, feed_df, light_df, atpasa_df]
)
inv_df = pd.merge(inv_df, sensors_df, on=['event_date', 'locus_group_id'], how='left')

# inv_df.to_csv('data/demand/cam_invplus_20240115.csv', index=False)
inv_df.info()

inv_df = pd.read_csv('cam_smolt_quality/data/demand/cam_invplus_20240115.csv')
inv_df['event_date'] = pd.to_datetime(inv_df['event_date'], utc=True)

inv_df = pd.merge(inv_df, atpasa_df, on=keys, how='left')

from scipy.stats import ttest_ind


group_a_sites = ['Ahoni', 'Johnson 2']
group_b_sites = ['Pilpilehue', 'Loncochalgua', 'Marilmo']

inv_df.loc[inv_df['final_site_name'].isin(group_a_sites), 'site_group'] = 'A'
inv_df.loc[inv_df['final_site_name'].isin(group_b_sites), 'site_group'] = 'B'

site_groups = inv_df['site_group'].unique()
grouped = inv_df.groupby('site_group')
feature_names = [col for col in inv_df.columns if col not in keys and col not in ['year_class', 'fg_order', 'final_lp_id', 'locus_group_id', 'final_site_name', 'site_group' ,'light_regime:type_name']]


g1 = 'A'
g2 = 'B'
results = []
for feature in feature_names:
    group1 = grouped.get_group(g1)[feature]
    group2 = grouped.get_group(g2)[feature]
    
    group1 = group1[group1.notna()]
    group2 = group2[group2.notna()]

    t_stat, p_val = ttest_ind(group1, group2)
    results.append({
        'feature_name': feature,
        'group:1': g1,
        'group:2': g2,
        'P-Value': p_val,
        'group1_len': len(group1),
        'group2_len':len(group2)
    })

# Convert results to DataFrame for easier viewing
results_df = pd.DataFrame(results)
results_df.to_csv('cam_smolt_quality/data/demand/t-test_per_feature.csv', index=False)

    
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 15 17:58:43 2024

# @author: dmitrii
# """

# df = pd.read_excel('C:/Users/dmitrii/Downloads/20241001-CAM-Operculum-dataset.csv')

# df = df.drop([col for col in df.columns if '.' in col],axis=1)
# df.to_csv('C:/Users/dmitrii/Downloads/20241001-CAM-Operculum-dataset.csv')


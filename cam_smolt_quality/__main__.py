import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils

from cam_smolt_quality.configs.cam_config import config

# req_sites = pd.DataFrame([
#         ['Pilpilehue', '2022-07-01', '2022-08-31'],
#         ['Loncochalgua', '2022-09-01','2022-09-30'],
#         ['Marilmo', '2022-10-01','2022-10-31'],
#         ['Ahoni', '2022-11-01','2022-11-30'],
#         ['Johnson 2', '2023-01-01','2023-01-31'],
#     ], 
#     columns=['site_name', 'from_date', 'to_date']
# )
# req_sites['from_date'] = pd.to_datetime(req_sites['from_date'], utc=True)
# req_sites['to_date'] = pd.to_datetime(req_sites['to_date'], utc=True)

# stock_df = extr.extract_data('stockings_mrts', config)
# stock_df = stock_df[stock_df['to_site_name'].isin(req_sites['site_name'].tolist())]

# g_dfs = []
# for g_id, g_df in stock_df.groupby('to_site_name'):
#     r_df = req_sites[req_sites['site_name']==g_id]
#     g_df = g_df[((g_df['transfer_date']>=r_df['from_date'].item()) & (g_df['transfer_date']<=r_df['to_date'].item()))]
#     g_dfs.append(g_df)
# stock_df = pd.concat(g_dfs, axis=0).reset_index(drop=True)

# add_args = {'DST_LPS': tuple(stock_df['from_lp_id'].unique().tolist())}
# movements_ratio_df = extr.extract_data('movements_ratio_mrts', config, additional_args=add_args)
    
# add_args = {'LP_IDS': tuple(movements_ratio_df['historic_lp_id'].unique().tolist())}
# keys = ['event_date', 'site_name', 'locus_id', 'locus_name', 'lp_id', 'fish_group']

# ltlg_df = extr.extract_data('locus_to_locus_group_mrts', config)
# inv_df = extr.extract_data('inventory_mrts', config, additional_args=add_args)

# final_lps = dict(zip(movements_ratio_df['historic_lp_id'], movements_ratio_df['final_lp_id']))
# inv_df['final_lp_id'] = inv_df['lp_id'].replace(final_lps)
# ltlg_dict = dict(zip(ltlg_df['locus_id'], ltlg_df['locus_group_id']))
# inv_df['locus_group_id'] = inv_df['locus_id'].replace(ltlg_dict)
# inv_df = pd.merge(inv_df, stock_df[['from_lp_id', 'to_site_name']].rename({'from_lp_id': 'final_lp_id', 'to_site_name': 'final_site_name'}, axis=1), on='final_lp_id', how='left')

# atpasa_df = extr.extract_data('lab_atpasa_mrts', config, additional_args=add_args)
# atpasa_df = utils.add_prefix(atpasa_df, keys=keys, prefix_name='atpasa')

# mortality_df = extr.extract_data('mortality_mrts', config, additional_args=add_args)
# rsns = mortality_df.groupby(['mortality_reason'])['mortality_count'].sum().sort_values(ascending=False).index[:10].tolist()
# mortality_df = mortality_df[mortality_df['mortality_reason'].isin(rsns)].pivot_table(index=keys, columns='mortality_reason', values='mortality_count', aggfunc='sum').reset_index()
# mortality_df = utils.add_prefix(mortality_df, keys=keys, prefix_name='mortality')

# treatment_df = extr.extract_data('treatments_mrts', config, additional_args=add_args)
# treatment_df = treatment_df.pivot_table(index=keys, columns='active_substance_name', values='amount', aggfunc='sum').reset_index()
# treatment_df = utils.add_prefix(treatment_df, keys=keys, prefix_name='active_substance_name')

# feed_df = extr.extract_data('feed_mrts', config, additional_args=add_args)
# feed_df = feed_df.groupby(keys)['feed_amount'].sum().reset_index()
# feed_df = utils.add_prefix(feed_df, keys=keys, prefix_name='feed')

# jobs_df = extr.extract_data('jobs_test_mrts', config, skip_full_format=True)
# logbook_df = extr.extract_data('logbook_mrts', config)

# locus_df = inv_df[['locus_id', 'locus_name']].drop_duplicates()
# sensors_df = pd.concat([logbook_df, jobs_df], axis=0)
# sensors_df = sensors_df.drop_duplicates(keep='last') #Keep jobs data
# sensors_df = sensors_df[sensors_df['variable'].notna()]
# # del(jobs_df, logbook_df)
# sensors_df = sensors_df.pivot_table(values='value', columns=['variable'], index=['event_date', 'locus_group_id'], aggfunc='mean')
# sensors_df = utils.add_prefix(sensors_df, keys='', prefix_name='sensor').reset_index()

# light_df = extr.extract_data('light_regime_mrts', config, additional_args=add_args)
# light_df = utils.add_prefix(light_df, keys=keys, prefix_name='light_regime')

# inv_df = reduce(
#     lambda left, right: pd.merge(left, right, on=keys, how='left'),
#     [inv_df, mortality_df, treatment_df, feed_df, light_df]
# )
# inv_df = pd.merge(inv_df, sensors_df, on=['event_date', 'locus_group_id'], how='left')

# inv_df.to_csv('data/demand/cam_invplus_20240115.csv', index=False)
# atpasa_df.to_csv('data/demand/cam_atpasa_20240115.csv', index=False)
# inv_df.info()

#inv_df = pd.merge(inv_df, atpasa_df, on=keys, how='outer')


# # -*- coding: utf-8 -*-
# """
# Created on Mon Jan 15 17:58:43 2024

# @author: dmitrii
# """

# df = pd.read_excel('C:/Users/dmitrii/Downloads/20241001-CAM-Operculum-dataset.csv')

# df = df.drop([col for col in df.columns if '.' in col],axis=1)
# df.to_csv('C:/Users/dmitrii/Downloads/20241001-CAM-Operculum-dataset.csv')



transfer_mortality_df = extr.extract_data('mortality_transfer_mrts', config)
stock_df = extr.extract_data('stockings_mrts', config)

# dtarget_df = pd.read_csv('D:\\projects\\ecto\\operators\\operator_smolt_quality\cam_smolt_quality\\notebooks\\data\\targets.csv')

# r_df = transfer_mortality_df[transfer_mortality_df['locus_id'].isin(stock_df['locus_id'].unique().tolist())]

stock_df = stock_df.rename({'to_site_name': 'site_name', 'to_locus_id': 'locus_id'}, axis=1)
n_days = 90
keys = ['site_name', 'locus_id', 'fish_group']
data_df = []
for tr_id, tr_df in tqdm(stock_df.groupby(keys)):
    res_d = dict(zip(keys, tr_id))
    
    g_df = transfer_mortality_df.copy()
    for key in keys:
        g_df = utils.base_filter(g_df, col=key, name=res_d[key])
    res_d['locus_len'] = len(g_df)

    last_day = g_df['event_date'].min() + pd.Timedelta(days=n_days)
    sum_mrts = g_df.loc[g_df['event_date'] <= last_day, 'mortality_count'].sum()
    target_value = sum_mrts/tr_df['stock_cnt'].sum() * 100
    res_d[f'mrtperc_first_{n_days}d'] = target_value
    data_df.append(res_d)
data_df = pd.DataFrame(data_df)
data_df['mrtperc_first_90d'].plot.hist(bins=100)

mean = data_df['mrtperc_first_90d'].mean()
rate = 1/mean
var = 1/(rate**2)
pdf = rate*np.exp(-rate*mean)

p = 0.977
x = -np.log(1-p)/rate

np.mean(data_df['mrtperc_first_90d'] > x)

# dtarget_df.columns
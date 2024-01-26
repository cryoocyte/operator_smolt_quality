import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils

from cam_smolt_quality.configs.cam_config import config

import logging
utils.init_logging()
logger = logging.getLogger(__name__)


stock_df = extr.extract_data('stockings_mrts', config)
inv_df = extr.extract_data('inventory_mrts', config)
mortality_df = extr.extract_data('mortality_mrts', config)

keys = ['to_site_name', 'to_fish_group']
stock_df = stock_df.groupby(keys).agg(
    min_transfer_date=('transfer_date', 'min'),
    max_transfer_date=('transfer_date', 'max'),
    stock_cnt=('stock_cnt', 'sum'),
).reset_index()



cycles_df = inv_df.groupby(['fish_group', 'site_name', 'site_type']).agg(
    min_date=('event_date', 'min'),
    max_date=('event_date', 'max'),
    max_fish_cnt=('start_fish_cnt', 'max'),
    max_fish_wg=('fish_wg', lambda x: x.quantile(0.95))
).reset_index()
cycles_df['days_length'] = (r_df['max_date'] - r_df['min_date']).dt.days.astype(int)




stock_df = stock_df.rename({'to_site_name': 'site_name', 'to_fish_group': 'fish_group'}, axis=1)
n_days = 90
keys = ['site_name', 'fish_group']
data_df = []
for tr_id, tr_df in tqdm(stock_df.groupby(keys)):
    res_d = dict(zip(keys, tr_id))
    # if res_d['fish_group'] == '56G2301.SNLCY2309.N.INV':
    #     break
    
    g_df = mortality_df.copy()
    for key in keys:
        g_df = utils.base_filter(g_df, col=key, name=res_d[key])

    last_day = g_df['event_date'].min() + pd.Timedelta(days=n_days)
    sum_mrts = g_df.loc[g_df['event_date'] <= last_day, 'mortality_cnt'].sum()
    target_value = sum_mrts/tr_df['stock_cnt'].item() * 100
    res_d['mortality_cnt'] = sum_mrts
    res_d['stock_cnt'] = tr_df['stock_cnt'].item()
    res_d[f'mrtperc_first_{n_days}d'] = target_value
    data_df.append(res_d)
stock_df = pd.DataFrame(data_df)
stock_df = pd.merge(stock_df, cycles_df, on=['site_name', 'fish_group'], how='left')

stock_df['days_length'].plot.hist(bins=50)

sns.scatterplot(data=stock_df, x='days_length', y='mrtperc_first_90d')

data_df['mrtperc_first_90d'].plot.hist(bins=100)

mean = data_df['mrtperc_first_90d'].mean()
rate = 1/mean
var = 1/(rate**2)
pdf = rate*np.exp(-rate*mean)

p = 0.977
x = -np.log(1-p)/rate

np.mean(data_df['mrtperc_first_90d'] > x)

# dtarget_df.columns
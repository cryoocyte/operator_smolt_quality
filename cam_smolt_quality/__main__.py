import pandas as pd
import numpy as np
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils
import ecto_ds.procedural.connect as connect
import os

from cam_smolt_quality.configs import CURRENT_DATE, PIPELINE_TYPE, WRITE_PARAMS, ROOT_DIR
from cam_smolt_quality.configs.cam_config import config, START_DATE, LICENSEE_ID, MRTPERC_GROUPS
import cam_smolt_quality.modules.gen as gen
import logging
utils.init_logging()
logger = logging.getLogger(__name__)
logger.info(f'Starting {PIPELINE_TYPE} pipeline...')






#Extract
movements_df = extr.extract_data('movements_mrts', config)
stock_df = extr.extract_data('stockings_mrts', config)
sw_sites = tuple(stock_df['to_site_name'].unique().tolist())
sw_inv_df = extr.extract_data('sw_inventory_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites}, add_spec_replacer=True)
fw_inv_df = extr.extract_data('fw_inventory_mrts', config, add_spec_replacer=True)

fg_df = extr.extract_data('cam_fishgroups', config)
sw_mortality_df = extr.extract_data('sw_mortality_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites}, add_spec_replacer=True)
fw_mortality_df = extr.extract_data('fw_mortality_mrts', config, add_spec_replacer=True)
luf_fish_df = extr.extract_data('lab_luf_fish', config)

vaccines_df = extr.extract_data('vaccines_mrts', config)
atpasa_df = extr.extract_data('lab_atpasa_mrts', config)
treatments_df = extr.extract_data('treatments_mrts', config)
fw_light_df = extr.extract_data('light_regime_mrts', config, add_spec_replacer=True)

sw_feed_df = extr.extract_data('sw_feed_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites}, add_spec_replacer=True)
fw_feed_df = extr.extract_data('fw_feed_mrts', config, add_spec_replacer=True)

sensors_df = extr.extract_data('sensors_mrts', config)
# sensors_df = sensors_df[sensors_df['variable'].notna()].reset_index(drop=True)

logbook_df = extr.extract_data('logbook_mrts', config)
jobs_df = extr.extract_data('jobs_mrts', config)
locus_locus_group_df = extr.extract_data('locus_to_locus_group_mrts', config)
locus_to_fish_group_df = extr.extract_data('locus_to_fish_group', config, add_spec_replacer=True)

site_map_df = extr.extract_data('site_map', config)
fish_group_map_df = extr.extract_data('fish_groups_map', config)

#Dataset
stock_df = gen.get_raw_target(stock_df, sw_mortality_df, sw_inv_df, sw_feed_df)

if PIPELINE_TYPE == 'forecast':
    FORECAST_VALUES_THRESHOLD = {
        'fish_wg': 130,
        'atpasa': 17    
    }
    active_inv_df = fw_inv_df[fw_inv_df['site_name']=='UPS']
    active_inv_df = active_inv_df[active_inv_df['event_date'] >= pd.to_datetime(CURRENT_DATE, utc=True) - pd.Timedelta(days=5)]
    active_fw_df = pd.merge(
        active_inv_df.groupby(['fish_group', 'site_name'])['fish_wg'].max().reset_index(),
        atpasa_df.groupby(['fish_group', 'site_name'])['atpasa'].max().reset_index(),
    how='left')
    mask = ((active_fw_df['fish_wg'] >= FORECAST_VALUES_THRESHOLD['fish_wg']) | (active_fw_df['atpasa'] >= FORECAST_VALUES_THRESHOLD['atpasa']))
    active_fw_df = active_fw_df[mask]
    active_inv_df = active_inv_df[active_inv_df['fish_group'].isin(active_fw_df['fish_group'].unique())]
    active_inv_df = active_inv_df.groupby(['event_date', 'fish_group'])[['end_fish_cnt', 'end_fish_bms']].sum().reset_index()
    max_fw_date = active_inv_df['event_date'].max()
    active_inv_df = active_inv_df[active_inv_df['event_date'] == max_fw_date]
    active_inv_df = active_inv_df.set_index('fish_group')
    
    active_fw_df['is_transfer'] = False
    active_fw_df = active_fw_df.set_index('fish_group')
    active_fw_df['smolt_wg'] = active_inv_df['end_fish_bms']/active_inv_df['end_fish_cnt'] * 1000
    active_fw_df['min_transfer_date'] = max_fw_date
    active_fw_df = active_fw_df.drop(['fish_wg', 'atpasa'],axis=1).reset_index()
    active_fw_df['days_length'] = 0
    
    stock_df = stock_df[stock_df['is_transfer']]
    stock_df['is_transfer'] = True
    stock_df = pd.concat([stock_df, active_fw_df],axis=0)
    
else:
    stock_df = gen.target_full_analysis(stock_df, use_all=True, draw=True)
fg_paths_df = gen.get_fg_paths(movements_df, stock_df)
fg_df = gen.prep_fishgroups(fg_df)
vaccines_df = gen.prep_vaccines(vaccines_df)
fw_feed_df = gen.get_feed(fw_feed_df)
jobs_df = gen.join_logbook_jobs(logbook_df, jobs_df)
jobs_df, jobs_q_df = gen.get_jobs(jobs_df, fw_inv_df, locus_locus_group_df)

#Sensors work
prefix_map = {'Hatchery': 'H', 'UPS': 'UPS', 'Fry': 'F', 'Ongrowing': 'OG'}
sensors_df['fw_locus_prefix'] = sensors_df['sensor_name'].map(lambda x: ''.join(prefix_map.get(v, '') for v in x.split()))
sensors_df = sensors_df[sensors_df['fw_locus_prefix'] != ''].reset_index(drop=True)
sensors_df = gen.sensor_data_clean(sensors_df)

stock_df = gen.construct_dataset(
    stock_df, fg_paths_df,fg_df, fw_inv_df,
    sw_inv_df, vaccines_df, fw_feed_df,
    fw_mortality_df, treatments_df, atpasa_df,
    jobs_df, jobs_q_df, luf_fish_df, sensors_df, fw_light_df
)
stock_df.to_csv(os.path.join(ROOT_DIR, f'data/modeling/stock_df_{PIPELINE_TYPE}.csv'), index=False)





 

import pandas as pd
import os
import numpy as np
from cam_smolt_quality.configs import READ_PARAMS, CURRENT_DATE, PIPELINE_TYPE, WRITE_PARAMS, ROOT_DIR
from cam_smolt_quality.configs.cam_config import config, LICENSEE_ID
import catboost as cb
from cam_smolt_quality.modules.utils import resid_func
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.connect as connect
import ecto_ds.procedural.utils as utils
import shap

import logging
utils.init_logging()
logger = logging.getLogger(__name__)
logger.info(f'Starting {PIPELINE_TYPE} pipeline...')

def calculate_quantile_for_series(series, value):
    sorted_series = series.sort_values()
    count_less_equal = sorted_series[sorted_series <= value].count()
    quantile = count_less_equal / len(sorted_series)
    return quantile

CONFIDENCE_LINEAR_RESID_PARAMS = {
    'nSFR_90d': {
        'z_score': 1.959963984540054,
        'params': [0.04358936, -0.00958266],
    },
    'mrtperc_90d:all': {
        'z_score': 1.959963984540054,
        'params': [0.30585913, 0.07583746],
    },
}
MODEL_ID = {'nSFR_90d': 113, 'mrtperc_90d:all': 114}
SHAP_TOP_N_FEATURES = 5

def commit_forecasts(forecast_df, table_name="ecto_prod.dst.clc_smolt_quality_forecast"):
    
    logger.info("-- Inserting to %s table" % table_name)
    objects = [tuple(int(item) if isinstance(item, np.integer) else item for item in row) for i, row in forecast_df.iterrows()]
    sql = f"""
        INSERT INTO {table_name}
            ({', '.join([col for col in forecast_df.columns])})
        VALUES %s 
        ON CONFLICT
            (forecast_date, transfer_date, site_id, fish_group_id, model_id, licensee_id, prediction_type) 
        DO NOTHING RETURNING 
            id;    
    """ 
    connect.commit(sql, objects, WRITE_PARAMS)
    
def commit_factors(factors_df, table_name="ecto_prod.dst.clc_smolt_quality_forecast"):
    pass
    
def get_increment(table_name='ecto_prod.dst.clc_smolt_quality_forecast'):
    query = f"""
        SELECT MAX(id) FROM {table_name};
    """
    result = connect.get_data_from_db(query, READ_PARAMS)['max'].item()
    start_forecast_id = 0 if pd.isnull(result) else result
    return start_forecast_id

#Stock from forecast pipeline
stock_df = pd.read_csv(os.path.join(ROOT_DIR, f'data/modeling/stock_df_{PIPELINE_TYPE}.csv'))

site_map_df = extr.extract_data('site_map', config)
fish_group_map_df = extr.extract_data('fish_groups_map', config)

#SHAP
past_df = pd.read_csv(os.path.join(ROOT_DIR, 'historical_data.csv'))

#FORECASTS
for target_name in ['nSFR_90d', 'mrtperc_90d:all']:
    model = cb.CatBoostRegressor()
    model.load_model(os.path.join(ROOT_DIR, f'data/modeling/{target_name}_v0.cbm'))
    stock_df[[f for f in model.feature_names_ if f not in stock_df.columns]] = np.nan
    conf_params = CONFIDENCE_LINEAR_RESID_PARAMS[target_name]    

    stock_df['prediction_result'] = model.predict(stock_df[model.feature_names_])
    forecast_df = stock_df[['min_transfer_date', 'fish_group', 'site_name', 'prediction_result', 'is_transfer', target_name]].reset_index(drop=True)
    forecast_df.rename({target_name: 'current_actual_result'}, axis=1)
    
    forecast_df['forecast_date'] = CURRENT_DATE
    forecast_df['model_id'] = MODEL_ID[target_name]
    forecast_df['licensee_id'] = LICENSEE_ID
    forecast_df['prediction_type'] = 'baseline_v0'
    
    forecast_df['confidence_p'] = conf_params['z_score']
    forecast_df['confidence_value'] = conf_params['z_score'] * resid_func(stock_df['prediction_result'], *conf_params['params'])
    

    start_forecast_id = get_increment() + 1
    forecast_df['id'] = np.arange(start_forecast_id, len(forecast_df)+start_forecast_id)

    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(stock_df[model.feature_names_])
    
    factors_df = []
    for i, (fg, row) in enumerate(stock_df.iterrows()):
        sv = pd.DataFrame(shap_values[i].values, columns=['shap_value',], index=shap_values.feature_names)
        sv.index.name = 'feature_name'
        sv = sv.reset_index().sort_values('shap_value', ascending=False)
        sv[ 'addition'] = np.nan
        sv.loc[sv['shap_value'] < 0, 'addition'] = 'decrease_target'
        sv.loc[sv['shap_value'] > 0, 'addition'] = 'increase_target'
        sv = pd.concat([
            sv[sv['addition']=='increase_target'].iloc[:SHAP_TOP_N_FEATURES],
            sv[sv['addition']=='decrease_target'].iloc[-SHAP_TOP_N_FEATURES:]
        ], axis=0)
        sv = sv.drop(['addition'],axis=1)
        
        sv['shap_perc'] = sv['shap_value']/(shap_values.base_values[i]+sv['shap_value'].abs()) * 100
        sv['fish_group'] = row['fish_group']
        sv['feature_value'] = shap_values[i].data[[shap_values.feature_names.index(c) for c in sv['feature_name']]]
        sv['feature_quantile'] = [calculate_quantile_for_series(past_df[f['feature_name']], f['feature_value']) for j,f in sv[['feature_name', 'feature_value']].iterrows()]
        sv.loc[sv['feature_value'].isna(), 'feature_quantile'] = np.nan
        factors_df.append(sv)
    factors_df = pd.concat(factors_df,axis=0).reset_index(drop=True)
    factors_df = pd.merge(
        factors_df, forecast_df[['fish_group', 'id']].rename({'id': 'forecast_id'},axis=1),
        on=['fish_group'],
        how='left'
    ).drop(['fish_group'],axis=1).rename({'id': 'forecast_id'}, axis=1)
    
    forecast_df = pd.merge(forecast_df, site_map_df,  on='site_name').drop(['site_name'], axis=1)
    forecast_df = pd.merge(forecast_df, fish_group_map_df, on='fish_group').drop(['fish_group'], axis=1)
    forecast_df = forecast_df.rename({target_name: 'current_actual_result', 'min_transfer_date': 'transfer_date'}, axis=1)
    
    
    commit_forecasts(forecast_df)
    commit_factors(factors_df)



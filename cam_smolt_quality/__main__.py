import pandas as pd
import seaborn as sns
import catboost as cb

from tqdm import tqdm
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils
import ecto_ds.procedural.connect as connect

import ecto_ds.procedural.metrics as metric_funcs
from cam_smolt_quality.configs import CURRENT_DATE, PIPELINE_TYPE, WRITE_PARAMS, ROOT_DIR
from cam_smolt_quality.configs.cam_config import config, START_DATE, LICENSEE_ID, MRTPERC_GROUPS
import cam_smolt_quality.modules.gen as gen
import itertools 


import logging
utils.init_logging()
logger = logging.getLogger(__name__)
logger.info(f'Starting {PIPELINE_TYPE} pipeline...')


#Dummies
def get_dummies(data_df, feature_name, drop=True, prefix='', prefix_sep=':'):
    data_df = data_df.copy()
    if feature_name in data_df.columns:
        df = pd.get_dummies(data_df[feature_name], prefix=prefix, prefix_sep=prefix_sep, dtype='int')
        data_df = pd.concat([data_df, df], axis=1)
        if drop:
            data_df = data_df.drop([feature_name],axis=1,errors='ignore')
    else:
        pass
    return data_df

 
def get_proper_samples(df, p=0.10, input_type='test', random_state=111):
    df['transfer_group'] = df.groupby(['site_name', 'transfer_year']).ngroup()
    avg_group_size = df.groupby(['transfer_group']).size().mean()
    groups = df['transfer_group'].sample(int(p/avg_group_size * len(df)), random_state=random_state)
    idxs = df[df['transfer_group'].isin(groups.values.tolist())].index
    mask = df.index.isin(idxs)
    n_df = df[mask].reset_index(drop=True)
    n_df['input_type'] = input_type
    return n_df, mask

def split(data_df, random_state=111, test_p=0.10, val_p=0.10):
    test_df, mask = get_proper_samples(data_df, input_type='test', p=test_p, random_state=random_state)
    trainval_df = data_df[~mask].reset_index(drop=True)
    val_df, mask = get_proper_samples(trainval_df, input_type='val', p=val_p)
    train_df = trainval_df[~mask].reset_index(drop=True)
    train_df['input_type'] = 'train'
    data_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    return data_df

def backtesting_split(test_yrs=1, val_yrs=1):
    test_last_date = pd.to_datetime(CURRENT_DATE, utc=True) - pd.Timedelta(days=int(365 * test_yrs))
    val_last_date = test_last_date - pd.Timedelta(days=int(365 * val_yrs))
    data_df['input_type'] = 'train'
    data_df.loc[(data_df['min_transfer_date'] >= val_last_date), 'input_type'] = 'val'
    data_df.loc[(data_df['min_transfer_date'] >= test_last_date), 'input_type'] = 'test'
    return data_df

def commit(forecast_df):
    table = "ecto_prod.dst.clc_smolt_quality_forecast"
    logger.info("-- Inserting to %s table" % table)
    objects = [tuple(int(item) if isinstance(item, np.integer) else item for item in row) for i, row in forecast_df.iterrows()]
    sql = f"""
        INSERT INTO {table}
            ({', '.join([col for col in forecast_df.columns])})
        VALUES %s 
        ON CONFLICT
            (forecast_date, transfer_date, site_id, fish_group_id, model_id, licensee_id, prediction_type) 
        DO NOTHING RETURNING 
            id;    
    """ 
    connect.commit(sql, objects, WRITE_PARAMS)

#Modeling
import shap
from eli5.sklearn import PermutationImportance
import optuna
from sklearn.metrics import mean_squared_error
import ecto_ds.procedural.math as math

def train_evaluate_model(data_df, feature_names, target_name, verbose=0, evaluate=True, log_transform=False, early_stopping_rounds=None, seed=123,  *args, **kwargs):
    train_idxs = data_df['input_type'] == 'train'
    val_idxs = data_df['input_type'] == 'val'
    
    if log_transform:
        log_target_name = f'log:{target_name}'
        log_eps = 1e-4
        data_df[log_target_name] = np.log(data_df[target_name] + log_eps)

    model = cb.CatBoostRegressor(*args, **kwargs, random_seed=seed)
    model.fit(
        data_df.loc[train_idxs if evaluate else (train_idxs | val_idxs), feature_names],
        data_df.loc[train_idxs if evaluate else (train_idxs | val_idxs), log_target_name if log_transform else target_name],
        eval_set=(
            data_df.loc[val_idxs, feature_names],
            data_df.loc[val_idxs, log_target_name if log_transform else target_name],
        ) if evaluate else None,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose
    )
  
    data_df['prediction'] = model.predict(data_df[feature_names])
    if log_transform:
        data_df['prediction'] = np.exp(data_df['prediction'] + log_eps)
    smape_metrics = 100 - data_df.groupby('input_type')[[target_name, 'prediction']].apply(lambda x:  metric_funcs.smape(x[target_name],x['prediction'])).rename('1-smape')
    r2_metrics = data_df.groupby('input_type')[[target_name, 'prediction']].apply(lambda x: metric_funcs.r_squared(x[target_name],x['prediction'])).rename('r2')
    mae_metrics = data_df.groupby('input_type')[[target_name, 'prediction']].apply(lambda x: metric_funcs.mae(x[target_name],x['prediction'])).rename('mae')

    metrics = pd.concat([smape_metrics, r2_metrics, mae_metrics], axis=1)
    return data_df, model, metrics

def train_pipeline(data_df, feature_names, target_name, cvg_threshold, corr_threshold, fillna_values, log_transform, perm_selection=False, n_hyper_trials=50, seed=123):
    
    np.random.seed(seed)

    #Coverage threshold
    cvgs = data_df[feature_names].notna().mean(0)
    feature_names = cvgs[cvgs >= cvg_threshold].index.tolist()
    if fillna_values:
        for f in feature_names:
            m = data_df[f].median()
            data_df[f] = data_df[f].fillna(m)
            data_df[f] = data_df[f].replace({-np.inf:m, np.inf:m})
        
    corr_df  = data_df[feature_names].corr()
    cols_to_remove = set()
    for i in range(len(corr_df.columns)):
        for j in range(i):
            if abs(corr_df.iloc[i, j]) > corr_threshold:
                col_name = corr_df.columns[i]
                if ':sin' not in col_name and ':cos' not in col_name:
                    cols_to_remove.add(col_name)
    for col in cols_to_remove:
        feature_names.remove(col)
    
    if perm_selection:
        param_grid = {
            'learning_rate': [0.1,],
            'depth': [4,],
            'l2_leaf_reg': [0, 10, 100],
            'loss_function': ['Quantile', 'RMSE', 'Poisson', 'MAE']
        }
        param_combs = list(itertools.product(*param_grid.values()))
        data_df = backtesting_split()
        
        val_idxs = data_df['input_type'] == 'val'
        pi_dfs = []
        for i, params in tqdm(enumerate(param_combs), 'Permutation importance search...'):
            params = dict(zip(param_grid.keys(), params))
        
            data_df, model, metrics = train_evaluate_model(
                data_df, feature_names, target_name,
                verbose=0, evaluate=True, early_stopping_rounds=50, log_transform=log_transform, seed=seed, **params)
            perm = PermutationImportance(model, cv="prefit").fit(
                data_df.loc[val_idxs, feature_names],
                data_df.loc[val_idxs,target_name]
            ) 
            
            pi_df = pd.DataFrame({ 'feature_name': feature_names,'pi': perm.feature_importances_,})
            # pi_df['fold'] = k
            pi_dfs.append(pi_df)
        pi_dfs = pd.concat(pi_dfs, axis=0)
        # sns.boxplot(data=pi_dfs, x='pi', y='feature_name', showfliers=False)
        pi_df = pi_dfs.groupby(['feature_name'])['pi'].mean().sort_values(ascending=False).reset_index()
        cyclic_features = [f for f in pi_df['feature_name'] if ':sin' in f or ':cos' in f]
        pi_df = pi_df[((pi_df['pi'] >= 0) | pi_df['feature_name'].isin(cyclic_features))]
        feature_names = pi_df['feature_name'].tolist()
        
    # data_df = split(data_df, random_state=111, val_simple=True)
    def objective(trial):
        # Define hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        depth = trial.suggest_int('depth', 4, 8)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0, 100)
        border_count = trial.suggest_int('border_count', 25, 255)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 50)
        loss_function = trial.suggest_categorical('loss_function', ['MAE', 'RMSE', 'MAPE', 'Quantile', 'LogLinQuantile', 'Poisson'])
    
        #Create and fit the model
        _, model, metrics = train_evaluate_model(
            data_df, feature_names, target_name,
            verbose=0, 
            log_transform=log_transform, 
            learning_rate=learning_rate, 
            depth=depth, 
            n_estimators=1000, 
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            min_data_in_leaf=min_data_in_leaf,
            loss_function=loss_function, 
            evaluate=True,
            early_stopping_rounds=50
        )
    
        # Calculate the validation loss
        score = metrics.loc['val', 'r2']/metrics.loc['val', 'mae']
        
        #Setup best iteration
        trial.set_user_attr("n_estimators", model.get_best_iteration())

        return score
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=seed),
        direction='maximize'
    )
    # optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=n_hyper_trials)
    
    best_params = study.best_params
    n_estimators = study.best_trial.user_attrs.get("n_estimators", None)
    best_params['n_estimators'] = n_estimators
    data_df, model, metrics = train_evaluate_model(
        data_df, feature_names, target_name, evaluate=False,
        log_transform=log_transform,
        **best_params
    )

    if 'nSFR' in target_name:
        data_df['prediction:bucket'] = 'Low'
        data_df.loc[data_df['prediction']>=1,'prediction:bucket'] = 'High'
        test_df = data_df[data_df['input_type']=='test']
        clf_cm_df, clf_report_df = metric_funcs.create_confusion_matrix_and_metrics(test_df[f'{target_name}:bucket'], test_df['prediction:bucket'], classes=data_df[f'{target_name}:bucket'].unique().tolist())
    elif 'mrtperc' in target_name:
        data_df['prediction:bucket'] = pd.cut(data_df['prediction'], bins=list(MRTPERC_GROUPS.values())+[100,], labels=list(MRTPERC_GROUPS.keys()))
        test_df = data_df[data_df['input_type']=='test']
        clf_cm_df, clf_report_df = metric_funcs.create_confusion_matrix_and_metrics(test_df[f'{target_name}:bucket'], test_df['prediction:bucket'], classes=data_df[f'{target_name}:bucket'].unique().tolist())
    else:
       clf_cm_df, clf_report_df = None 
          
    return data_df, model, metrics, clf_cm_df, clf_report_df , best_params


def compute_influence(X_train, y_train, model):
    influences = []
    original_mse = mean_squared_error(y_train, model.predict(X_train))

    for i in tqdm(range(len(X_train)), 'Calculating Cook`s distances...'):
        # Exclude the current data point
        X_train_new = np.delete(X_train, i, axis=0)
        y_train_new = np.delete(y_train, i, axis=0)

        # Clone the model to avoid refitting the same model
        new_model = cb.CatBoostRegressor(**model.get_params())
        new_model.fit(X_train_new, y_train_new, verbose=0)

        # Compute new MSE without the data point
        new_mse = mean_squared_error(y_train_new, new_model.predict(X_train_new))

        # Estimate the influence
        influence = new_mse - original_mse
        influences.append(influence)

    return influences


def get_fi(model, features=[]):
    fi_res = pd.DataFrame(features, columns=["feature"])
    fi_res["importances"] = model.feature_importances_
    fi_res = fi_res.sort_values("importances", ascending=False)
    return fi_res


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

sw_feed_df = extr.extract_data('sw_feed_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites})
fw_feed_df = extr.extract_data('fw_feed_mrts', config)

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


# for g_id, g_df in stock_df.groupby('site_name'):
#     break
#     fig, ax = plt.subplots()
#     g_df.plot.scatter('min_transfer_date','nSFR_90d', ax=ax, zorder=2)
#     ax.set_ylim(0.5, 1.5)
#     ax.axhline(1, ls='--', color='lightgrey', zorder=1)
    
    
if PIPELINE_TYPE == 'forecast':
    stock_df = stock_df[stock_df['is_transfer']]
else:
    stock_df = gen.target_full_analysis(stock_df, use_all=True, draw=False)
fg_paths_df = gen.get_fg_paths(movements_df, stock_df)
fg_df = gen.prep_fishgroups(fg_df)
vaccines_df = gen.prep_vaccines(vaccines_df)
fw_feed_df = gen.get_feed(fw_feed_df)
jobs_df = gen.join_logbook_jobs(logbook_df, jobs_df)
jobs_df, jobs_q_df = gen.get_jobs(jobs_df, fw_inv_df, locus_locus_group_df)


#Sensors work
prefix_map = {'Hatchery': 'H', 'UPS': 'UPS', 'Fry': 'F', 'Ongrowing': 'OG'}
sensors_df['fw_locus_prefix'] = sensors_df['sensor_name'].map(lambda x: ''.join(prefix_map.get(v, '') for v in x.split()))

#pH
mask = (sensors_df['sensor_type_name'] == 'PH') & ((sensors_df['value'] < 5.5) | (sensors_df['value'] > 8.5))
sensors_df = sensors_df[~mask].reset_index(drop=True)

stock_df = gen.construct_dataset(stock_df, fg_paths_df,fg_df, fw_inv_df, sw_inv_df, vaccines_df, fw_feed_df, fw_mortality_df, treatments_df, atpasa_df, jobs_df, jobs_q_df)
target_name = 'nSFR_90d' #'mrtperc_90d:all' #

#Dummies
if PIPELINE_TYPE == 'forecast':
    data_df = stock_df.copy()
else:
    data_df = stock_df[~stock_df['is_transfer']].reset_index(drop=True)

data_df = get_dummies(data_df, feature_name='fg:strain_name', prefix='fg:strain_name')
# data_df = get_dummies(data_df, feature_name='calendar:transfer_month', prefix='calendar:transfer_month')
for prefix in ['H', 'F', 'OG', 'UPS']: #'I', 
    # data_df = get_dummies(data_df, feature_name=f'calendar:month:{prefix}', prefix=f'calendar:season:{prefix}')
    data_df = utils.sincos_transform(data_df, f'calendar:dayofyear:{prefix}')
    data_df = data_df.drop(f'calendar:dayofyear:{prefix}', axis=1)
data_df = utils.sincos_transform(data_df, 'calendar:transfer_dayofyear')
data_df = data_df.drop('calendar:transfer_dayofyear', axis=1)


if PIPELINE_TYPE == 'forecast':
    def predict(): pass
    model = cb.CatBoostRegressor()
    model.load_model('cam_smolt_quality/data/models/model_nsfr_v0.cbm')
    
    data_df[[f for f in model.feature_names_ if f not in data_df.columns]] = np.nan
    
    MODEL_ID = 113
    data_df['prediction_result'] = model.predict(data_df[model.feature_names_])
    forecast_df = data_df[['min_transfer_date', 'fish_group', 'site_name', 'prediction_result', target_name]].reset_index(drop=True)
    forecast_df.rename({target_name: 'current_actual_result'},axis=1)
    forecast_df['forecast_date'] = CURRENT_DATE
    forecast_df['model_id'] = MODEL_ID
    forecast_df['licensee_id'] = LICENSEE_ID
    forecast_df['prediction_type'] = 'baseline_v0'
    forecast_df = pd.merge(forecast_df, site_map_df, on='site_name').drop(['site_name'], axis=1)
    forecast_df = pd.merge(forecast_df, fish_group_map_df, on='fish_group').drop(['fish_group'], axis=1)
    forecast_df = forecast_df.rename({target_name: 'current_actual_result', 'min_transfer_date': 'transfer_date'},axis=1)
    
    commit(forecast_df)
    
    
# Cycle lengths
data_df['transfer_len'].plot.hist(bins=30)
plt.title('Transfer length in days at Site/Fish group level')
plt.xlabel('Days')
plt.tight_layout()

# #nSFR charts
# fig, ax = plt.subplots(1, 1, figsize=(8,4), sharey=True)
# stock_df['feed:nSFR:overall'].plot.hist(bins=40, ax=ax)
# ax.set_title('feed:nSFR:overall')
# stock_df['feed:nSFR:positive_rate'].plot.hist(bins=40, ax=ax[1])
# ax[1].set_title('feed:nSFR:positive_rate')
# plt.tight_layout()

#Sensor chartss
# for scol in sensor_cols:    
#     fig, ax = plt.subplots(1, figsize=(12,5))
#     cols = [c for c in stock_df.columns if scol in c and 'max' in c]
#     r_df = stock_df[cols].melt()
#     sns.boxplot(data=r_df, x='value', y='variable', ax=ax)
#     plt.tight_layout()

#Treatment charts
# uniqs = treatments_df['active_substance_name'].unique().tolist()
# fig, ax = plt.subplots(4, len(uniqs), figsize=(15,9), sharey=True)
# for i,tm in enumerate(uniqs):
#     stock_df[f'treatment:days_in:{tm}'].plot.hist(bins=40,ax=ax[0][i])
#     ax[0][i].set_title(f'{tm}')
#     ax[0][i].set_ylim(0, 50)
# ax[0][0].set_ylabel('Number of days')
# for i,tm in enumerate(uniqs):
#     stock_df[f'treatment:consecutive_without:{tm}'].plot.hist(bins=40,ax=ax[1][i])
#     # ax[1][i].set_title(f'{tm}')
#     ax[1][i].set_ylim(0, 50)
# ax[1][0].set_ylabel('Consecutive days without')
# for i,tm in enumerate(uniqs):
#     stock_df[f'treatment:days_since_last:{tm}'].plot.hist(bins=40,ax=ax[2][i])
#     # ax[2][i].set_title(f'{tm}')
#     ax[2][i].set_ylim(0, 50)
# ax[2][0].set_ylabel('Days since last')
# for i,tm in enumerate(uniqs):
#     stock_df[f'treatment:amount_per_day:{tm}'].plot.hist(bins=40,ax=ax[3][i])
#     # ax[2][i].set_title(f'{tm}')
#     ax[3][i].set_ylim(0, 50)
# ax[3][0].set_ylabel('Amount per day')
# plt.tight_layout()


# Atpasa charts
# fig, ax = plt.subplots(1, 2, figsize=(12,4), sharey=True)
# stock_df['atpasa:latest_month'].plot.hist(bins=30, ax=ax[0])
# ax[0].set_title('mean latest_month')
# stock_df['atpasa:derivative'].plot.hist(bins=30, ax=ax[1])
# ax[1].set_title('derivative')
# fig.suptitle('Atpasa')
# plt.tight_layout()

#Light regime charts
# fig, ax = plt.subplots(1,4, figsize=(13,4), sharey=True)
# for i, name in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
#     stock_df[f'light:{name}:cvg_perc'].plot.hist(ax=ax[i])
#     ax[i].set_title(f'{name}')
# fig.suptitle('FW Light regime cvg percs')
# plt.tight_layout()

#Stages charts
# fig, ax = plt.subplots(1,5, figsize=(13,4), sharey=True)
# for i, fw_prefix in enumerate(['I', 'H', 'F', 'OG', 'UPS']):
#     stock_df[f'cycle:len:{fw_prefix}'].plot.hist(ax=ax[i])
#     ax[i].set_title(f'Stage: {fw_prefix}')
# fig.suptitle('FW Stages lengths')
# plt.tight_layout()

#Time sin/cos


# palette = {'January': '#FF6347',
#  'February': '#FF4500',
#  'March': '#90EE90',
#  'April': '#32CD32',
#  'May': '#9ACD32',
#  'June': '#87CEFA',
#  'July': '#1E90FF',
#  'August': '#40E0D0',
#  'September': '#FFDAB9',
#  'October': '#FFA500',
#  'November': '#FF8C00',
#  'December': '#FF7F7F'}
# month_mapping = {
#     1: "January", 2: "February",  3: "March", 4: "April",5: "May",6: "June",7: "July", 8: "August",
#     9: "September", 10: "October",  11: "November",12: "December"
# }
# data_df['transfer_month'] = data_df['transfer_month'].replace(month_mapping)
# sns.scatterplot(data=data_df.sort_values('transfer_month'), s=60,
#                 x='calendar:transfer_dayofyear:sin', y='calendar:transfer_dayofyear:cos',
#                 hue='transfer_month', hue_order=list(month_mapping.values()),
#                 palette=palette, zorder=2)
# # plt.legend(loc='upper right')
# plt.xlabel('calendar day sin'); plt.ylabel('calendar day cos')
# plt.axvline(0, ls='--', color='lightgrey', zorder=1); plt.axhline(0, ls='--', color='lightgrey', zorder=1)
# plt.title('Transfer day of year feature transformation.')
# plt.tight_layout()
# plt.show()


#Remove outliers
lw, uw = math.get_outliers(data_df, target_name)
data_df = data_df[((data_df[target_name] > lw) & (data_df[target_name] < uw))].reset_index(drop=True)

cvg_threshold = 0.3
corr_threshold = 0.8
fillna_values = True
log_transform = False

final_grid_results = []

#Features    
feature_groups = ['calendar:', 'sensor:', 'atpasa:', 'vaccine:', 'cycle:','fg:', 'temperature:', 'feed:', 'mortality:', 'light:', 'treatment:']
feature_names = [col for col in data_df.columns for fg in feature_groups if fg in col and 'log' not in col]


if 'nSFR' in target_name:
    data_df = backtesting_split()
    data_df[f'{target_name}:bucket'] = 'Low'
    data_df.loc[data_df[target_name]>=1,f'{target_name}:bucket'] = 'High'
elif 'mrtperc' in target_name:
    data_df = backtesting_split()
    # data_df = split(data_df, random_state=111, test_p=0.15, val_p=0.15)
    data_df[f'{target_name}:bucket'] = pd.cut(data_df[target_name], bins=list(MRTPERC_GROUPS.values())+[100,], labels=list(MRTPERC_GROUPS.keys()))
else:
    pass
    
# data_df['input_type'].value_counts()
# fig, ax = plt.subplots(2,1,figsize=(13,7))
# sns.boxplot(data=data_df, x=target_name, y='input_type',  ax=ax[0], zorder=2)
# sns.scatterplot(data=data_df, x='min_transfer_date', y=target_name, hue='input_type',  ax=ax[1], zorder=2)
# fig.suptitle(f'CAM Smolt Quality. {target_name}: train/test split.')
# plt.tight_layout()

data_df, model, metrics, clf_cm_df, clf_report_df, model_params = train_pipeline(
    data_df, feature_names, target_name,
    cvg_threshold, corr_threshold, fillna_values,
    log_transform, perm_selection=True, n_hyper_trials=700, seed=111
)
# model.save_model('cam_smolt_quality/data/models/model_v1.cbm', format="cbm")
fi_res = get_fi(model, features=model.feature_names_).reset_index(drop=True)


# TP  TN  FP  FN   F-score       FPR
# Low    8   5   1   3  0.800000  0.166667
# High   5   8   3   1  0.714286  0.272727

#               1-smape        r2       mae
# input_type                               
# test        95.762086  0.124356  0.080130
# train       98.578318  0.919552  0.028041
# val         98.674583  0.899754  0.025561

# mask = data_df['input_type'] != 'test'
# cooks = compute_influence(X_train=data_df.loc[mask, model.feature_names_].values, y_train=data_df.loc[mask, target_name].values, model=model)
# # data_df.loc[mask, 'cook'] = cooks

x = np.linspace(data_df[target_name].min()-1e-3, data_df[target_name].max(), 20)
y = x
fig, ax = plt.subplots(1, 2, figsize=(8,4),) # sharey=True
sns.scatterplot(data=data_df.loc[data_df['input_type'].isin(['train', 'val'])], x=target_name, y='prediction', hue='input_type', palette={'train': 'tab:green', 'val': 'tab:blue'}, ax=ax[0], zorder=2)
if 'nSFR' in target_name:
    # ax[0].axvline(0, ls='--'); ax[0].axhline(1, ls='--');
    ax[1].axvline(1, ls='--'); ax[1].axhline(1, ls='--');
ax[0].plot(x, y, ls='--', color='lightgrey', zorder=1)
ax[0].set_title('Train/Val data')
sns.scatterplot(data=data_df.loc[data_df['input_type']=='test'], x=target_name, y='prediction', color='tab:red',ax=ax[1], zorder=2)
ax[1].plot(x, y, ls='--', color='lightgrey', zorder=1)
ax[1].set_title('Test data')
fig.suptitle(f'{target_name}; Actual to Prediction plot')
plt.tight_layout()
plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_df[model.feature_names_])
explanation = shap.Explanation(values=shap_values, data=data_df, feature_names=model.feature_names_)

#SHAP boxplots
shap_values = explainer(data_df[model.feature_names_])
shap_i = pd.DataFrame({
    'feature': model.feature_names_,
    'shap_importance':  np.abs(shap_values.values).mean(axis=0)
}).sort_values('shap_importance', ascending=False).reset_index(drop=True)

shap.summary_plot(shap_values, data_df[model.feature_names_], plot_size=(12, 8))
# plt.xlim(-0.2,0.2)
plt.title(f'{target_name}. Baseline SHAP values')
plt.tight_layout()

res_df = data_df.sort_values(target_name)
ordered_features = shap_i.iloc[:10]['feature'].tolist()
baseline_value = data_df[target_name].median()#shap_values.base_values[0]



# shap_values = ...
plt.ioff()
for sample_idx in range(res_df.shape[0]): 
    features_to_show = len(ordered_features)
    other_features_str = f'Other {shap_values.shape[1]-features_to_show} features'
    actual = res_df[target_name].iloc[sample_idx]
    pred = res_df['prediction'].iloc[sample_idx]

    # group_color = groups_palette[res_df['target_group'].iloc[sample_idx]]
    
    sample_data = shap_values.data[sample_idx]
    sample_values = shap_values.values[sample_idx]
    data_type  = res_df['input_type'].iloc[sample_idx]
    # target_group = res_df['target_group'].iloc[sample_idx]
    features_str = [c for c in ordered_features]
    feature_yticks = dict(zip(range(features_to_show), features_str))
    feature_yticks[features_to_show] = other_features_str
    
    fig, ax = plt.subplots(1, 2, figsize=(14,6), sharey=True, gridspec_kw={'width_ratios': [1, 2]})

    ax[1].axvline(baseline_value, ls='--', color='lightgrey', zorder=1, label='Average/Baseline mortality')
    ax[1].axvline(actual, ls='--', color='tab:green', zorder=2, label='Actual')
    ax[1].axvline(pred, ls='--', color='tab:purple', zorder=1, label='Prediction')

    pos_value = float(baseline_value)
    features_box = []
    for plot_idx, feature in enumerate(ordered_features + [other_features_str,]):
        if feature == other_features_str:
            sample_value = 0
            for c in shap_values.feature_names:
                if c not in ordered_features:
                    sample_value += sample_values[shap_values.feature_names.index(c)]
            features_box.append({'feature': feature, 'value': np.nan})
        else:
            feature_idx = shap_values.feature_names.index(feature)
            sample_value = sample_values[feature_idx]
            features_box.append({'feature': feature, 'value': sample_data[[shap_values.feature_names.index(feature)]].item()})
            
        ax[1].arrow(
            pos_value, plot_idx, 
            sample_value, 0,
            width=0.9,
            head_width=0.9,
            head_length=0,#0.10*abs(sample_value), 
            length_includes_head=True,
            color='tab:red' if sample_value > 0 else 'tab:blue', zorder=2
        )
        annotation_pos = (pos_value + sample_value + sample_value * 0.1, plot_idx)
        ax[1].annotate(
            f'{sample_value:.4f}', annotation_pos, color='black', ha='left' if sample_value > 0 else 'right', va='center',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.0'))
    
        pos_value += sample_value
    ax[1].set_yticks(list(feature_yticks.keys()))
    ax[1].set_yticklabels(list(feature_yticks.values()))   
    ax[1].set_xlim(data_df[target_name].min()-data_df[target_name].min()*0.1, data_df[target_name].max()+data_df[target_name].max()*0.1)
    ax[1].set_xlabel(target_name)
    ax[1].invert_yaxis()
    
    features_box = pd.DataFrame(features_box)

    r_df = data_df[ordered_features].copy()
    features_box = pd.merge(features_box, r_df.min().rename('min').reset_index().rename({'index': 'feature'},axis=1))
    features_box = pd.merge(features_box, r_df.max().rename('max').reset_index().rename({'index': 'feature'},axis=1))
    features_box['value'] = (features_box['value']-features_box['min'])/(features_box['max']-features_box['min'])
    
    r_df = ((r_df - r_df.min()))/(r_df.max()-r_df.min())
    r_df = r_df.melt(value_vars=ordered_features)
        
    sns.stripplot(data=r_df, x='value', y='variable', ax=ax[0], zorder=1, color='lightgrey')
    sns.scatterplot(data=features_box, x='value', y='feature', ax=ax[0], color='tab:green', zorder=1)
    ax[0].set_xlabel('Normalized Feature value')
    #ax[0].axvline(0, linestyle='--', color='darkgrey')
    
    site_name = res_df.iloc[sample_idx]["site_name"]
    fig.suptitle(f'Site: {site_name}. Fish group: {res_df.iloc[sample_idx]["fish_group"]}. Input type: {data_type}')
    # plt.show()
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    import os
    wd = os.path.join(ROOT_DIR, f'data/figures/waterfall/')
    if not os.path.exists(wd):
        os.makedirs(wd)
    plt.savefig(os.path.join(wd, f"{sample_idx}_{res_df.index[sample_idx]}.jpg"))
    plt.close()
    

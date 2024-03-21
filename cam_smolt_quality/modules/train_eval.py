import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from scipy.stats import norm
from scipy import stats
import matplotlib.dates as mdates
import statsmodels.api as sm

import shap
import random
from sklearn.inspection import permutation_importance
import catboost as cb
import optuna
from sklearn.metrics import mean_squared_error
import ecto_ds.procedural.math as math
from sklearn.model_selection import KFold
import ecto_ds.procedural.utils as utils
import ecto_ds.procedural.metrics as metric_funcs
from cam_smolt_quality.configs import CURRENT_DATE, PIPELINE_TYPE, WRITE_PARAMS, ROOT_DIR

import logging

utils.init_logging()

logger = logging.getLogger(__name__)
logger.info(f'Starting {PIPELINE_TYPE} pipeline...')


def get_proper_samples(df, p=0.10, input_type='test', random_state=111):
    df['transfer_group'] = df.groupby(['site_name', 'transfer_year']).ngroup()
    avg_group_size = df.groupby(['transfer_group']).size().mean()
    groups = df['transfer_group'].sample(int(p / avg_group_size * len(df)), random_state=random_state)
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

def onemonth_crossout_split(df):
    df['month'] = df['min_transfer_date'].dt.strftime('%Y-%m')
    months = sorted(df['month'].unique())

    index_list = []
    for month in months:
        month_indexes = df[df['month'] == month].index
        other_months_indexes = df[df['month'] != month].index
        index_tuple = (month, month_indexes, other_months_indexes)
        index_list.append(index_tuple)

    return index_list

def oneyear_backtestin_split(data_df, test_yrs=1):
    data_df['min_transfer_date'] = pd.to_datetime(data_df['min_transfer_date'], utc=True)
    test_last_date = data_df['min_transfer_date'].max() - pd.Timedelta(days=int(365 * test_yrs))
    data_df['input_type'] = 'train'
    data_df.loc[(data_df['min_transfer_date'] >= test_last_date),'input_type'] = 'test'
    return data_df

def onemonth_backtesting_split(data_df, test_months=1):
    data_df['min_transfer_date'] = pd.to_datetime(data_df['min_transfer_date'], utc=True)
    test_last_date = data_df['min_transfer_date'].max() - pd.Timedelta(days=int(30 * test_months))
    data_df['input_type'] = 'train'
    data_df.loc[(data_df['min_transfer_date'] >= test_last_date), 'input_type'] = 'test'
    return data_df

def onemonth_rolling_split(data_df):
    data_df['min_transfer_date'] = pd.to_datetime(data_df['min_transfer_date'], utc=True)
    data_df['month'] = data_df['min_transfer_date'].dt.strftime("%Y-%m")
    unique_months = data_df['month'].unique()
    sorted_months = sorted(unique_months)
    split_indexes = []
    for i in range(1, len(sorted_months)):
        train_months = sorted_months[:i]
        test_month = sorted_months[i]
        train_indexes = data_df[data_df['month'].isin(train_months)].index
        test_indexes = data_df[data_df['month'] == test_month].index
        split_indexes.append((train_indexes, test_indexes))
    return split_indexes

def oneyear_crossout_split(df):
    df['year'] = df['min_transfer_date'].dt.year
    years = sorted(df['year'].unique())
    index_list = []
    for year in years:
        year_indexes = df[df['year'] == year].index
        other_years_indexes = df[df['year'] != year].index
        index_tuple = (year, year_indexes, other_years_indexes)
        index_list.append(index_tuple)
    return index_list


def select_features(train_df, test_df, feature_names, target_name, evaluate=True, seed=123, num_features_to_select=10, *args, **kwargs):
    model = cb.CatBoostRegressor(
        *args, **kwargs, random_seed=seed, task_type=kwargs.get("task_type", "CPU"))

    if kwargs.get('select_features', True):
        summary = model.select_features(
            train_df[feature_names],
            train_df[target_name],
            eval_set=(
                test_df[feature_names],
                test_df[target_name],
            ) if evaluate else None,
            features_for_select=feature_names,
            algorithm='RecursiveByShapValues',
            logging_level=kwargs.get('logging_level', 'Silent'),
            train_final_model=False,
            num_features_to_select=num_features_to_select,
        )
        optimized_feature_names = summary['selected_features_names']
        return optimized_feature_names
    
def select_features_cv(df, feature_names, evaluate=True, seed=123, num_features_to_select=10, *args, **kwargs):
    all_features = []
    indexes = onemonth_crossout_split(df)
    for fold in tqdm(indexes, 'Feature selection...'):
        year, train_index, val_index = fold
        train_df = df.iloc[train_index]
        val_df = df.iloc[val_index]
        
        optimized_feature_names = select_features(
            train_df, val_df, feature_names, target_name, evaluate=evaluate, seed=seed,
            num_features_to_select=num_features_to_select,
            *args, **kwargs
        )
        all_features.append(optimized_feature_names)
    optimized_feature_names = list({el for f in all_features for el in f})
    return optimized_feature_names


def train_evaluate_model_cv(df, feature_names, target_name, n_splits=5, num_features_to_select=10, *args, **kwargs):
    # kf = KFold(
    #     n_splits=n_splits, shuffle=True,
    #     random_state=kwargs.get('seed', 123)
    # )
    metrics_list = []
    indexes = onemonth_crossout_split(df)

    for fold in indexes:
        year, train_index, val_index = fold
        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)
        try:
            model, metrics = train_evaluate_model(
                train_df,
                val_df,
                optimized_feature_names, target_name, evaluate=True,
                *args, **kwargs
            )
            metrics_list.append(metrics)
        except cb.CatBoostError:
            pass
    metrics_cv = pd.concat(metrics_list, axis=0)
    return model, metrics_cv


def train_evaluate_model(train_df, val_df, feature_names, target_name, evaluate=True, verbose=0, early_stopping_rounds=None, seed=64, *args, **kwargs):
    model = cb.CatBoostRegressor(
        *args, **kwargs,
        random_seed=seed,
        task_type=kwargs.get("task_type", "CPU")
    )

    model.fit(
        train_df[feature_names],
        train_df[target_name],
        eval_set=(
            val_df[feature_names],
            val_df[target_name],
        ) if evaluate else None,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )
    train_df['input_type'] = 'train'
    if val_df is not None:
        val_df['input_type'] = 'val'
        res_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    else:
        res_df = train_df.copy()
    res_df['prediction'] = model.predict(res_df[feature_names])
    res_df.loc[res_df['prediction'] <= 0, 'prediction'] = train_df[target_name].min()
    
    metrics = res_df.groupby('input_type')[[target_name, 'prediction']].apply(
        lambda x: pd.Series({
            '1-smape': 100 - metric_funcs.smape(x[target_name], x['prediction']),
            'r2': metric_funcs.r_squared(x[target_name], x['prediction']),
            'mae': metric_funcs.mae(x[target_name], x['prediction']),
            'rmse': metric_funcs.rmse(x[target_name], x['prediction']),

        })
    )
    return model, metrics


def tune_pipeline(df, test_df, feature_names, target_name, n_hyper_trials=50, seed=123, save_study=True, verbose=1):

    study_path = os.path.join(ROOT_DIR, f'optuna_study_{target_name}.pkl')
    def objective(trial):
        # Define hyperparameters to be tuned
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        depth = trial.suggest_int('depth', 4, 6)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 0, 50)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 10)
        loss_function = trial.suggest_categorical('loss_function', ['MAE', 'RMSE', 'Quantile', 'LogLinQuantile', 'Poisson'])
        n_estimators = trial.suggest_int('n_estimators', 50, 200)

        # Create and fit the model
        max_month = df['month'].max()
        train_df = df[df['month'] != max_month].reset_index(drop=True)
        val_df = df[df['month'] == max_month].reset_index(drop=True)
        
        model, metrics_cv = train_evaluate_model(
            train_df, val_df, #test_df
            feature_names,
            target_name,
            verbose=0,
            learning_rate=learning_rate,
            depth=depth,
            n_estimators=n_estimators,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            loss_function=loss_function,
            evaluate=False,
            seed=seed
        )
        
        # r2 = metrics_cv.loc['val', 'r2'].mean()
        rmse = metrics_cv.loc['val','rmse'].mean()
        # trial.set_user_attr("n_estimators", model.best_iteration_)
        return rmse

    if os.path.exists(study_path):
        # Load the existing study
        study = load(study_path)
        logger.info(f"Loaded existing study from {study_path}")
    else:
        # Create a new study
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=seed),
            direction='minimize'
            # directions=["maximize", "minimize"]
        )
        logger.info("Initialized a new study")
    if verbose:  
        optuna.logging.enable_default_handler()
    else:
        optuna.logging.disable_default_handler()

    study.optimize(objective, n_trials=n_hyper_trials)
    if save_study:
        dump(study, study_path)

    best_trial = study.best_trial  # study.best_trials[0]
    best_params = best_trial.params
    best_params.update(best_trial.user_attrs)
    return best_params

def get_dummies(data_df, feature_name, drop=True, prefix='', prefix_sep=':'):
    data_df = data_df.copy()
    if feature_name in data_df.columns:
        df = pd.get_dummies(
            data_df[feature_name], prefix=prefix, prefix_sep=prefix_sep, dtype='int')
        data_df = pd.concat([data_df, df], axis=1)
        if drop:
            data_df = data_df.drop([feature_name], axis=1, errors='ignore')
    else:
        pass
    return data_df


def get_fi(model, features=[]):
    fi_res = pd.DataFrame(features, columns=["feature"])
    fi_res["importances"] = model.feature_importances_
    fi_res = fi_res.sort_values("importances", ascending=False)
    return fi_res

def get_final_results(data_df, model_params, feature_names, min_year=2021):
    scores_df = []
    split_indexes = onemonth_rolling_split(data_df)
    for train_indexes, test_indexes in tqdm(split_indexes, 'backtesting proceed...'):
        train_df = data_df.loc[train_indexes].reset_index(drop=True)
        test_df = data_df.loc[test_indexes].reset_index(drop=True)
        if test_df['min_transfer_date'].min().year < min_year:
            continue
    
        # model_params = tune_pipeline(
        #     train_df, test_df,
        #     optimized_feature_names, target_name,
        #     n_hyper_trials=100, seed=42, save_study=False, verbose=0
        # )
        model, metrics = train_evaluate_model(
            train_df, test_df,
            feature_names, target_name, evaluate=False,
            verbose=0, seed=42, 
            **model_params
        )
        test_df['prediction'] = model.predict(test_df[feature_names])
        
        scores_df.append({
            'split_month': test_df['month'].iloc[0],
            'feature_names': optimized_feature_names,
            'model_params': model_params,
            'baseline_value': train_df[target_name].mean(),
            'test_target': test_df[target_name],
            'test_prediction': test_df['prediction'],
            'test_mae': metrics.loc['val', 'mae'],
        })
    scores_df = pd.DataFrame(scores_df)
    
    backtest_df = pd.DataFrame({
        'split_month':scores_df['split_month'],
        target_name: scores_df['test_target'].explode(),
        'prediction': scores_df['test_prediction'].explode(),
        'baseline_value': scores_df['baseline_value']
    })
    backtest_df['input_type'] = 'test'
    backtest_df.loc[backtest_df['prediction'] <= 0, 'prediction'] = np.nan
    backtest_df['split_date'] = pd.to_datetime(backtest_df['split_month']+'-01', utc=True)
    
    dates_metrics_df = backtest_df.groupby('split_date')[[target_name, 'prediction']].apply(
        lambda x: pd.Series({
            '1-smape': 100 - metric_funcs.smape(x[target_name], x['prediction']),
            'mae': metric_funcs.mae(x[target_name], x['prediction']),
            'rmse': metric_funcs.rmse(x[target_name], x['prediction']),
            'resid': (x['prediction'] - x[target_name]).mean()
        })
    ).reset_index()
    # base_metrics_df = backtest_df.groupby('split_date')[[target_name, 'baseline_value']].apply(
    #     lambda x: pd.Series({
    #         'mae': 100 - metric_funcs.smape(x[target_name], x['baseline_value']),
    #         'rmse': metric_funcs.mae(x[target_name], x['baseline_value']),
    #         'resid': (x['baseline_value'] - x[target_name]).mean()
    #     })
    # ).reset_index()
    
    metrics = backtest_df.groupby('input_type')[[target_name, 'prediction']].apply(
        lambda x: pd.Series({
            '1-smape': 100 - metric_funcs.smape(x[target_name], x['prediction']),
            'r2': metric_funcs.r_squared(x[target_name], x['prediction']),
            'mae': metric_funcs.mae(x[target_name], x['prediction']),\
            'rmse': metric_funcs.rmse(x[target_name], x['prediction']),
        })
    )
    backtest_df['month'] = backtest_df['split_date'].dt.month
    month_metrics_df = backtest_df.groupby('month')[[target_name, 'prediction']].apply(
        lambda x: pd.Series({
            'mae': metric_funcs.mae(x[target_name], x['prediction']),
            'resid': (x['prediction'] - x[target_name]).mean()
        })
    ).reset_index()

        
    return model, backtest_df, dates_metrics_df, month_metrics_df, metrics

stock_df = pd.read_csv(os.path.join(ROOT_DIR, 'data/stock_df.csv'))
target_name = 'mrtperc_90d:all'  #'nSFR_90d' #

cvg_threshold = 0.3
corr_threshold = 0.8
fillna_values = True
var_threshold = 1e-3

data_df = stock_df[~stock_df['is_transfer']].reset_index(drop=True)
data_df = get_dummies(data_df, feature_name='fg:strain_name', prefix='fg:strain_name')
data_df = utils.sincos_transform(data_df, 'calendar:transfer_dayofyear')
data_df = data_df.drop('calendar:transfer_dayofyear', axis=1)

if 'nSFR' in target_name:
    data_df[f'{target_name}:bucket'] = 'Low'
    data_df.loc[data_df[target_name] >= 1, f'{target_name}:bucket'] = 'High'
# elif 'mrtperc' in target_name:
    # data_df = backtesting_split(data_df)
    # data_df = split(data_df, random_state=111, test_p=0.15, val_p=0.15)
#     data_df[f'{target_name}:bucket'] = pd.cut(data_df[target_name], bins=list(MRTPERC_GROUPS.values())+[100,], labels=list(MRTPERC_GROUPS.keys()))
# else:
#     pass


# Remove outliers
if PIPELINE_TYPE == 'train':
    if 'nSFR' in target_name:
        lw, uw = math.get_outliers(data_df, target_name)
    elif 'mrtperc' in target_name:
        lw, uw = 0, 10
    data_df = data_df[(((data_df[target_name] > lw) & (data_df[target_name] < uw)))].reset_index(drop=True)

# Features
feature_groups = ['calendar:', 'light:', 'lab:', 'sensor:', 'jobs:' 'atpasa:',
                  'vaccine:', 'cycle:', 'fg:', 'temperature:', 'feed:', 'mortality:', 'treatment:']
feature_names = [col for col in data_df.columns for fg in feature_groups if fg in col and 'log' not in col]

# Var threshold
feature_names = (data_df[feature_names].var() > var_threshold).index.tolist()

# Coverage threshold
cvgs = data_df[feature_names].notna().mean(0)
feature_names = cvgs[cvgs >= cvg_threshold].index.tolist()

# Fillna
if fillna_values:
    for f in feature_names:
        m = data_df[f].median()
        data_df[f] = data_df[f].fillna(m)
        data_df[f] = data_df[f].replace({-np.inf: m, np.inf: m})

corr_df = data_df[feature_names].corr()
cols_to_remove = set()
for i in range(len(corr_df.columns)):
    for j in range(i):
        if abs(corr_df.iloc[i, j]) > corr_threshold:
            col_name = corr_df.columns[i]
            if ':sin' not in col_name and ':cos' not in col_name:
                cols_to_remove.add(col_name)
for col in cols_to_remove:
    feature_names.remove(col)

# split_indexes = onemonth_rolling_split(data_df)
# for train_indexes, test_indexes in tqdm(split_indexes, 'backtesting proceed...'):
#     train_df = data_df.loc[train_indexes].reset_index(drop=True)
#     test_df = data_df.loc[test_indexes].reset_index(drop=True)
#     if test_df['min_transfer_date'].min().year < 2021:
#         continue
#     if len(test_df) > 1: break
# emp_df = data_df[data_df['min_transfer_date']>test_df['min_transfer_date'].max()]
# # data_df['input_type'].value_counts()
# fig, ax = plt.subplots(1,1,figsize=(13,4))
# sns.scatterplot(data=train_df, x='min_transfer_date', y=target_name, color='tab:green',  ax=ax, zorder=2)
# sns.scatterplot(data=test_df, x='min_transfer_date', y=target_name, color='tab:red',  ax=ax, zorder=2)
# sns.scatterplot(data=emp_df, x='min_transfer_date', y=target_name, color='tab:grey',  ax=ax, zorder=2)

# fig.suptitle(f'CAM Smolt Quality. {target_name}: train/test split.')
# plt.tight_layout()


# fs_results_df = []
# split_indexes = onemonth_rolling_split(data_df)
# for train_indexes, test_indexes in tqdm(split_indexes, 'backtesting proceed...'):
#     train_df = data_df.loc[train_indexes].reset_index(drop=True)
#     test_df = data_df.loc[test_indexes].reset_index(drop=True)
#     if test_df['min_transfer_date'].min().year < 2022:
#         continue
#     kf = KFold(n_splits=5, shuffle=True, random_state=123)
#     fcv_model_params = {
#         'learning_rate': 0.03, 'depth': 4,'n_estimators': 200,
#         'loss_function': 'RMSE', 'l2_leaf_reg': 10, 'min_data_in_leaf': 5,
#     }
#     pi_scores = []
#     for train_index, val_index in tqdm(kf.split(train_df), 'Feature selection CV... '):
#         model, metrics = train_evaluate_model(
#             train_df.iloc[train_index].reset_index(drop=True),
#             train_df.iloc[val_index].reset_index(drop=True),
#             feature_names=feature_names, 
#             target_name=target_name, evaluate=True,
#             verbose=0, seed=123, 
#             early_stopping_rounds=50,
#             **fcv_model_params
#         )
#         perm = permutation_importance(
#             model,  
#             train_df.iloc[val_index][feature_names],
#             train_df.iloc[val_index][target_name],
#             n_repeats=10,
#             random_state=42,
#         )
#         pi_scores.append(perm.importances_mean)
#     pi_scores = np.array(pi_scores)
#     pi_means = np.mean(pi_scores, axis=0)
#     pi_df = pd.DataFrame({'feature_name': feature_names, 'pi': pi_means}).sort_values('pi', ascending=False)
#     # optimized_feature_names = pi_df.loc[pi_df['pi'] > 0, 'feature_name']
#     fs_results_df.append(pi_df)  
# fs_results_df = pd.concat(fs_results_df, axis=0).reset_index(drop=True)
# res_df = fs_results_df.groupby('feature_name')['pi'].mean().reset_index()
# res_df.to_csv(os.path.join(ROOT_DIR, f'data/modeling/{target_name}_feautres.csv'), index=False)

res_df = pd.read_csv(os.path.join(ROOT_DIR, f'data/modeling/{target_name}_feautres.csv'))
optimized_feature_names = res_df.loc[res_df['pi'] > 0, 'feature_name'].tolist()
  

ALL_MODEL_PARAMS = {
    'nSFR_90d': {
        'n_estimators': 200, 'learning_rate': 0.03,
        'depth': 4, 'loss_function': 'MAE',
        'l2_leaf_reg': 1,  "min_data_in_leaf": 5,
        'subsample': 1
    },
    'mrtperc_90d:all': {
        'n_estimators': 200, 'learning_rate': 0.01,
        'depth': 4, 'loss_function': 'Poisson',
        'l2_leaf_reg': 10,  "min_data_in_leaf": 5,
        'subsample': 0.5
    },
}

# import itertools

# param_dict = {
#     'learning_rate': [0.01, 0.03],
#     'depth': [4,],
#     'n_estimators': [200,],
#     'loss_function': ['MAE', 'RMSE', 'Poisson'],
#     'l2_leaf_reg': [1, 10, 20, 50, 100],
#     'min_data_in_leaf': [5,],
#     'subsample': [0.5, 0.7, 1.0],
# }
# param_combinations = list(itertools.product(*param_dict.values()))
# param_dicts = [dict(zip(param_dict.keys(), combination)) for combination in param_combinations]
# print(f"Total number of combinations: {len(param_dicts)}")

# ALL_RESULTS = [] 
# for i, model_params in enumerate(param_dicts, start=1):
#     print(f"Combination {i}: {model_params}")

#     model, backtest_df, dates_metrics_df, month_metrics_df, metrics = get_final_results(data_df, model_params, optimized_feature_names, min_year=2021)
#     model_params['r2'] = metrics.loc['test', 'r2']
#     model_params['rmse'] = metrics.loc['test', 'rmse']

#     ALL_RESULTS.append(model_params)
# ALL_RESULTS = pd.DataFrame(ALL_RESULTS)

model_params = ALL_MODEL_PARAMS[target_name]
model, backtest_df, dates_metrics_df, month_metrics_df, metrics = get_final_results(
    data_df, model_params, optimized_feature_names, min_year=2021
)
backtest_df = backtest_df.sort_values('split_month')
# incr_scores_df = []
# for month in backtest_df['split_month'].unique():
#     filtered_df = backtest_df[backtest_df['split_month'] <= month]
#     try: r2 = metric_funcs.r_squared(filtered_df[target_name], filtered_df['prediction'])
#     except: r2= np.nan
#     incr_scores_df.append({
#         'split_month': month, 
#         'rmse': metric_funcs.rmse(filtered_df[target_name], filtered_df['prediction']),
#         'r2': r2
#     })
# incr_scores_df = pd.DataFrame(incr_scores_df)
# incr_scores_df.loc[incr_scores_df['r2'] < 0, 'r2'] = 0
# fig, ax = plt.subplots(2,1,figsize=(12,5), sharex=True)
# incr_scores_df.set_index('split_month')['rmse'].plot.bar(width=0.9, color='tab:brown',ax=ax[0], label='rmse')
# ax[0].set_ylabel('RMSE')
# incr_scores_df.set_index('split_month')['r2'].plot.bar(width=0.9, color='tab:blue',ax=ax[1], label='r2')
# ax[1].set_ylabel('R2')
# fig.suptitle(f'{target_name}: Model performance using test data over time (Incremental Scores)')
# plt.tight_layout()

if 'nSFR' in target_name:
    backtest_df['prediction:bucket'] = 'Low'
    backtest_df.loc[backtest_df['prediction'] >= 1, 'prediction:bucket'] = 'High'
    backtest_df[f'{target_name}:bucket'] = 'Low'
    backtest_df.loc[backtest_df[target_name] >= 1, f'{target_name}:bucket'] = 'High'
    clf_cm_df, clf_report_df = metric_funcs.create_confusion_matrix_and_metrics(
        backtest_df[f'{target_name}:bucket'],
        backtest_df['prediction:bucket'],
        classes=backtest_df[f'{target_name}:bucket'].unique().tolist()
    )

#### FINAL MODEL ####
final_model, metrics = train_evaluate_model(
    train_df=data_df,
    val_df=None,
    feature_names=optimized_feature_names, 
    target_name=target_name,
    evaluate=False,
    verbose=0,
    seed=42, 
    **model_params
)
final_model.save_model(os.path.join(ROOT_DIR, f'data/modeling/{target_name}_v0.cbm'))
data_df.to_csv(os.path.join(ROOT_DIR, 'historical_data.csv'), index=False)


data_df['prediction'] = final_model.predict(data_df[feature_names])
data_df = data_df.sort_values('prediction')
data_df['resid'] = data_df['prediction'] - data_df[target_name]
data_df['abs_resid'] = np.abs(data_df['resid'])

from scipy.optimize import curve_fit, minimize
from cam_smolt_quality.modules.utils import resid_func


def objective_func(params, x, y, lambda_reg):
    residuals = resid_func(x, *params) - y
    mse = np.mean(residuals**2)
    regularization = lambda_reg * sum([p**2 for p in list(params)])
    return mse + regularization

lambda_reg = {'nSFR_90d': 0.01, 'mrtperc_90d:all': 1}
result = minimize(objective_func, [1, 1], args=(data_df['prediction'], data_df['abs_resid'], lambda_reg[target_name]))
params = result.x
data_df['resid_smooth'] = resid_func(data_df['prediction'], *params)

confidence_level = 0.95
z_score = stats.norm.ppf((1 + confidence_level) / 2)



data_df['ci_lower'] = data_df['prediction'] - z_score * data_df['resid_smooth']
data_df['ci_upper'] = data_df['prediction'] + z_score * data_df['resid_smooth']
cvg = ((data_df['ci_lower'] <= data_df[target_name]) & (data_df['ci_upper'] >= data_df[target_name])).mean()

ax = data_df.plot.scatter('prediction', 'abs_resid', zorder=2)
data_df.plot('prediction', 'resid_smooth', zorder=2,ax=ax)
ax.axhline(0, ls='--', color='lightgrey', zorder=1)
ax.legend()
plt.tight_layout()
          
amin = data_df[target_name].min()
amax = data_df[target_name].max()

x = np.linspace(data_df[target_name].min() - 1e-3, data_df[target_name].max(), 20)
y = x
x0 = np.linspace(amin - 0.01*amin, amax + 0.01*amax, 200)
rs = resid_func(x0, *params)
cil = x0 - z_score * rs
ciu = x0 + z_score * rs


fig, ax = plt.subplots(1, 1, figsize=(7, 6), sharex=True)
if 'nSFR' in target_name:
    ax.axhline(1, ls='--')
    ax.axvline(1, ls='--')
ax.plot(y, x, ls='--', color='lightgrey', zorder=1)
sns.scatterplot(data=data_df, x=target_name, y='prediction', color='tab:red', ax=ax, zorder=2)
ax.plot(y, x, ls='--', color='lightgrey', zorder=1)
ax.set_title('T+1 final model results')
ax.set_ylim(amin, amax)
ax.set_xlim(amin, amax)

ax.fill_betweenx( 
    x0, cil, ciu, #data_df['prediction'], data_df['ci_lower'], data_df['ci_upper'],
    alpha=0.2, color='tab:red',
    label=f'{confidence_level*100}% Prediction Interval. Prediction interval: {cvg.round(2)*100}%'
)
ax.legend(loc='upper left')
fig.suptitle(f'{target_name}; Model summary')
plt.tight_layout()
plt.show()



explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(data_df[model.feature_names_])
explanation = shap.Explanation(
    values=shap_values, data=data_df, feature_names=model.feature_names_)

# SHAP boxplots
shap_values = explainer(data_df[model.feature_names_])
shap_i = pd.DataFrame({
    'feature': model.feature_names_,
    'shap_importance': np.abs(shap_values.values).mean(axis=0)
}).sort_values('shap_importance', ascending=False).reset_index(drop=True)

shap.summary_plot(
    shap_values, data_df[model.feature_names_], plot_size=(12, 8))
# plt.xlim(-0.2,0.2)
plt.title(f'{target_name}. Baseline SHAP values')
plt.tight_layout()


res_df = data_df.sort_values(target_name)
ordered_features = shap_i.iloc[:10]['feature'].tolist()
baseline_value = data_df[target_name].median()  # shap_values.base_values[0]
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
    data_type = res_df['input_type'].iloc[sample_idx]
    # target_group = res_df['target_group'].iloc[sample_idx]
    features_str = [c for c in ordered_features]
    feature_yticks = dict(zip(range(features_to_show), features_str))
    feature_yticks[features_to_show] = other_features_str

    fig, ax = plt.subplots(1, 2, figsize=(
        14, 6), sharey=True, gridspec_kw={'width_ratios': [1, 2]})

    ax[1].axvline(baseline_value, ls='--', color='lightgrey',
                  zorder=1, label='Average/Baseline mortality')
    ax[1].axvline(actual, ls='--', color='tab:green', zorder=2, label='Actual')
    ax[1].axvline(pred, ls='--', color='tab:purple',
                  zorder=1, label='Prediction')

    pos_value = float(baseline_value)
    features_box = []
    for plot_idx, feature in enumerate(ordered_features + [other_features_str,]):
        if feature == other_features_str:
            sample_value = 0
            for c in shap_values.feature_names:
                if c not in ordered_features:
                    sample_value += sample_values[shap_values.feature_names.index(
                        c)]
            features_box.append({'feature': feature, 'value': np.nan})
        else:
            feature_idx = shap_values.feature_names.index(feature)
            sample_value = sample_values[feature_idx]
            features_box.append({'feature': feature, 'value': sample_data[[
                                shap_values.feature_names.index(feature)]].item()})

        ax[1].arrow(
            pos_value, plot_idx,
            sample_value, 0,
            width=0.9,
            head_width=0.9,
            head_length=0,  # 0.10*abs(sample_value),
            length_includes_head=True,
            color='tab:red' if sample_value > 0 else 'tab:blue', zorder=2
        )
        annotation_pos = (pos_value + sample_value +
                          sample_value * 0.1, plot_idx)
        ax[1].annotate(
            f'{sample_value:.4f}', annotation_pos, color='black', ha='left' if sample_value > 0 else 'right', va='center',
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.0'))

        pos_value += sample_value
    ax[1].set_yticks(list(feature_yticks.keys()))
    ax[1].set_yticklabels(list(feature_yticks.values()))
    ax[1].set_xlim(data_df[target_name].min() - data_df[target_name].min()
                   * 0.1, data_df[target_name].max() + data_df[target_name].max() * 0.1)
    ax[1].set_xlabel(target_name)
    ax[1].invert_yaxis()

    features_box = pd.DataFrame(features_box)

    r_df = data_df[ordered_features].copy()
    features_box = pd.merge(features_box, r_df.min().rename(
        'min').reset_index().rename({'index': 'feature'}, axis=1))
    features_box = pd.merge(features_box, r_df.max().rename(
        'max').reset_index().rename({'index': 'feature'}, axis=1))
    features_box['value'] = (features_box['value'] - features_box['min']
                             ) / (features_box['max'] - features_box['min'])

    r_df = ((r_df - r_df.min())) / (r_df.max() - r_df.min())
    r_df = r_df.melt(value_vars=ordered_features)

    sns.stripplot(data=r_df, x='value', y='variable',
                  ax=ax[0], zorder=1, color='lightgrey')
    sns.scatterplot(data=features_box, x='value', y='feature',
                    ax=ax[0], color='tab:green', zorder=1)
    ax[0].set_xlabel('Normalized Feature value')
    # ax[0].axvline(0, linestyle='--', color='darkgrey')

    site_name = res_df.iloc[sample_idx]["site_name"]
    fig.suptitle(
        f'Site: {site_name}. Fish group: {res_df.iloc[sample_idx]["fish_group"]}. Input type: {data_type}')
    # plt.show()
    plt.legend(loc='upper right')
    plt.tight_layout()

    import os
    wd = os.path.join(ROOT_DIR, f'data/figures/waterfall/')
    if not os.path.exists(wd):
        os.makedirs(wd)
    plt.savefig(os.path.join(
        wd, f"{sample_idx}_{res_df.index[sample_idx]}.jpg"))
    plt.close()




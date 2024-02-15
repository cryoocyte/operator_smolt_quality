import pandas as pd
import seaborn as sns
from tqdm import tqdm
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import ecto_ds.procedural.extract as extr
import ecto_ds.procedural.utils as utils

from cam_smolt_quality.configs.cam_config import config, START_DATE
import cam_smolt_quality.modules.gen as gen

import logging
utils.init_logging()
logger = logging.getLogger(__name__)

def get_weighted_date(df, w):
    first_date = pd.to_datetime('1970-01-01', utc=True)
    int_date = (df['event_date'] - first_date).dt.days
    days = int((int_date * v_df[w]/v_df[w].sum()).sum())
    first_date = first_date + pd.Timedelta(days=days)
    return first_date

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

def smape(true, pred):
    res = 100/len(true) * np.sum(np.abs(pred - true) / (np.abs(true) + np.abs(pred)))
    return res

def get_proper_samples(df, p=0.10, input_type='test', random_state=111):
    df['transfer_group'] = df.groupby(['site_name', 'transfer_year']).ngroup()
    avg_group_size = df.groupby(['transfer_group']).size().mean()
    groups = df['transfer_group'].sample(int(p/avg_group_size * len(df)), random_state=random_state)
    idxs = df[df['transfer_group'].isin(groups.values.tolist())].index
    mask = df.index.isin(idxs)
    n_df = df[mask].reset_index(drop=True)
    n_df['input_type'] = input_type
    return n_df, mask

def split(data_df, random_state=111):
    test_df, mask = get_proper_samples(data_df, input_type='test', p=TEST_PERC, random_state=random_state)
    trainval_df = data_df[~mask].reset_index(drop=True)
    val_df, mask = get_proper_samples(trainval_df, input_type='val', p=VAL_PERC)
    train_df = trainval_df[~mask].reset_index(drop=True)
    train_df['input_type'] = 'train'
    data_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
    return data_df

def train_evaluate_model(data_df, feature_names, target_name, loss_function='MAE', verbose=0):
    train_idxs = data_df['input_type'] == 'train'
    val_idxs = data_df['input_type'] == 'val'
  
    model = cb.CatBoostRegressor(learning_rate=0.1, loss_function=loss_function)
    model.fit(
        data_df.loc[train_idxs, feature_names],
        data_df.loc[train_idxs, log_target_name if log_transform else target_name],
        eval_set=(
            data_df.loc[val_idxs, feature_names],
            data_df.loc[val_idxs, log_target_name if log_transform else target_name],
        ),
        early_stopping_rounds=100,
        verbose=verbose
    )
  
    data_df['prediction'] = model.predict(data_df[feature_names])
    if log_transform:
        data_df['prediction'] = np.exp(data_df['prediction'] + log_eps)
    metrics = 100 - data_df.groupby('input_type')[[target_name, 'prediction']].apply(lambda x: smape(x[target_name],x['prediction']))
    return data_df, model, metrics


#Extract
movements_df = extr.extract_data('movements_mrts', config)
stock_df = extr.extract_data('stockings_mrts', config)
sw_sites = tuple(stock_df['to_site_name'].unique().tolist())
sw_inv_df = extr.extract_data('sw_inventory_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites})
fw_inv_df = extr.extract_data('fw_inventory_mrts', config)

fg_df = extr.extract_data('cam_fishgroups', config)
sw_mortality_df = extr.extract_data('sw_mortality_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites})
fw_mortality_df = extr.extract_data('fw_mortality_mrts', config,)

vaccines_df = extr.extract_data('vaccines_mrts', config)
atpasa_df = extr.extract_data('lab_atpasa_mrts', config)
treatments_df = extr.extract_data('treatments_mrts', config)
fw_light_df = extr.extract_data('light_regime_mrts', config)

sw_feed_df = extr.extract_data('sw_feed_mrts', config, additional_args={'SITES_TO_INCLUDE': sw_sites})
fw_feed_df = extr.extract_data('fw_feed_mrts', config)


logbook_df = extr.extract_data('logbook_mrts', config)
jobs_df = extr.extract_data('jobs_test_mrts', config)

#Dataset
stock_df = gen.get_raw_target(stock_df, sw_mortality_df, sw_inv_df, sw_feed_df)
stock_df = gen.target_full_analysis(stock_df, draw=False)
fg_paths_df = gen.get_fg_paths(movements_df, stock_df)
fg_df = gen.prep_fishgroups(fg_df)
vaccines_df = gen.prep_vaccines(vaccines_df)

# fig, ax = plt.subplots(1,2, sharey=True)
# stock_df.plot.scatter('nSFR_90d:locus', 'TGC_90d',ax=ax[0], color='tab:orange')
# stock_df.plot.scatter('nSFR_positive_rate_90d', 'TGC_90d',ax=ax[1], color='tab:blue')

min_date = pd.to_datetime(START_DATE, utc=True)
s_dfs = []
r_dfs  = []
for (uniq_id, dst_fish_group), uniq_df in tqdm(fg_paths_df.groupby(['cycle_id', 'dst_fish_group'])):
    # break
    # if dst_fish_group == '68G1801.SNFAN1801.M.VER':
    #     break
    #Feature Groups
    uniq_fgs = uniq_df['fish_group'].tolist()
    uniq_fgs.remove(dst_fish_group)
    fw_df = fw_inv_df.loc[fw_inv_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
    sw_df = sw_inv_df.loc[sw_inv_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
    s_df = stock_df[stock_df['fish_group']==dst_fish_group].reset_index(drop=True)
    v_df = vaccines_df.loc[vaccines_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
    fe_df = fw_feed_df.loc[fw_feed_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
    mrt_df = fw_mortality_df.loc[fw_mortality_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
    l_df = fw_light_df.loc[fw_light_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        
    if sw_df['event_date'].min() == min_date or len(fw_df) == 0:
        s_dfs.append(s_df)
        continue
    #First dates
    first_feeding_date = fe_df['event_date'].min() #Simple, TODO!!!
    first_vaccine_date = get_weighted_date(v_df, 'vaccine:fish_cnt') #Weighted
    
    # sns.scatterplot(data=fw_df[fw_df['fw_locus_prefix'] == 'I'], x='event_date', y='fish_wg', hue='fish_group')
    # sns.scatterplot(data=fw_df, x='event_date', y='fish_wg', hue='fish_group')

    #   #Cycle features
    ## FW stages length
    fw_cycle_legnth = (fw_df['event_date'].max() - fw_df['event_date'].min()).days
    s_df['cycle:fw_legnth'] = fw_cycle_legnth
    fw_stages_sorted = ['I', 'H', 'F', 'OG', 'UPS']
    fw_df['fw_locus_prefix'] = fw_df['fw_locus_prefix'].astype('category').cat.set_categories(fw_stages_sorted)
    fw_dates = fw_df.groupby('fw_locus_prefix')['event_date'].agg(['min', 'max'])
    if fw_df['event_date'].min() == min_date:
        fw_dates.loc['I', 'min'] = np.nan
    if len(fw_dates) == 0:
        s_dfs.append(s_df)
        continue
    fw_dates = fw_dates.loc[[i for i in fw_stages_sorted if i in fw_dates.index]]
    fw_dates.loc['OG/UPS', 'min'] = fw_dates.loc[['OG', 'UPS'], 'min'].min()
    fw_dates.loc['OG/UPS', 'max'] = fw_dates.loc[['OG', 'UPS'], 'max'].max()
    fw_dates = fw_dates.drop(['OG', 'UPS'], axis=0)
    idx = 'OG/UPS'
    fw_max = fw_dates.loc[idx, 'max']
    fw_dates['max'] = fw_dates['min'].shift(-1)
    fw_dates.loc[idx, 'max'] = fw_max
    fw_stages_lengths = (fw_dates['max'] - fw_dates['min']).dt.days.astype(float)
    #Restrict to the latest I fish group
    if 'I' in fw_df['fw_locus_prefix'].unique():
        b_fg = fw_df[fw_df['fw_locus_prefix'] == 'I'].groupby('fish_group').ngroups
        fw_stages_lengths.loc['I'] = fw_stages_lengths.loc['I']//b_fg
    fw_stages_lengths.index = [f'cycle:len:{i}' for i in fw_stages_lengths.index]    

    for k in fw_stages_sorted:
        if k in ['OG', 'UPS']:
            k = 'OG/UPS'
        if f'cycle:len:{k}' in fw_stages_lengths.index: 
            if fw_stages_lengths[f'cycle:len:{k}'] == 0: res = np.nan
            else: res = fw_stages_lengths[f'cycle:len:{k}'] 
        else:  res = np.nan
        s_df[f'cycle:len:{k}'] = res

    fw_stages_season = fw_dates['min'].dt.month.apply(utils.season_from_month)
    fw_stages_season.index = [f'calendar:season:{i}' for i in fw_stages_season.index]    
    for k in fw_stages_sorted:
        if k in ['OG', 'UPS']:
            k = 'OG/UPS'
        if fw_stages_season[f'calendar:season:{k}'] == 'Undefined':
            s_df[f'calendar:season:{k}'] = np.nan
        else:
            s_df[f'calendar:season:{k}'] = fw_stages_season[f'calendar:season:{k}']
    
    #Weight on transfer
    wg_on_transfer = fw_df.loc[fw_df['event_date'] == s_df['min_date'].item(), 'fish_wg'].max()
    s_df['cycle:transfer_fish_weight'] = wg_on_transfer

    #Vaccines    
    #Define vaccine date/since days based on weighted fish cnt strategy
    days_since_vaccine = (s_df['min_trasnfer_date'].item() - first_vaccine_date).days
    s_df['vaccine:days_since'] = days_since_vaccine
    
    test = first_feeding_date - fw_dates.loc['H', 'min']
    s_df['test:date'] = test.days
    
    #Degree days sum
    degree_days_sum = fw_df['degree_days'].sum()
    s_df['temperature:fw_degree_days'] = degree_days_sum
    
    #Overall oSFR
    bms = fw_df['end_fish_bms'].max() - fw_df['end_fish_bms'].min()
    feed = fe_df['feed_amount'].sum()
    osfr = bms/feed * 100
    s_df['feed:oSFR_overall'] = osfr
    
    #Mortality rate since Fry stage
    mrts = mrt_df.loc[mrt_df['event_date'] > fw_dates.loc['H', 'max'], 'mortality_cnt'].sum()
    mrt_rate = mrts/(s_df['stock_cnt'].item()+mrts)
    s_df['mortality:mrt_rate_since_Fry'] = mrt_rate
    
    #Light regime (Photoperiod)
    l_df = l_df.groupby(['event_date', 'fw_locus_prefix'])['cnt'].sum().reset_index()
    # lyears = l_df[['year', 'season', ]].value_counts()/90 * 100 #'fw_locus_prefix'
    # lyears = lyears.iloc[:-1]
    min_date, max_date = l_df['event_date'].min(), l_df['event_date'].max()
    d_df = pd.date_range(min_date, max_date, freq='D').to_frame().rename({0: 'event_date'},axis=1)
    d_df = d_df.merge(l_df, on='event_date', how='outer')
    d_df['season'] = d_df['event_date'].dt.month.apply(utils.season_from_month)
    d_df['year'] = d_df['event_date'].dt.year
    # d_df['fw_locus_prefix'] = d_df['fw_locus_prefix'].ffill()
    d_df = d_df.groupby(['season', 'year'])['cnt'].apply(lambda x: x.notna().mean()).rename('cvg').reset_index()
    d_df = d_df.groupby('season')['cvg'].mean() * 100
    d_df.index = [f'light:{i}:cvg_perc' for i in d_df.index]    
    for k, v in d_df.items():
        s_df[k] = v
    
    #Vaccines used #TODO!!!
    # v_df = v_df.groupby(['event_date', 'site_name'])[[c for c in v_df.columns if 'prefix' in c]].any()
    # v_df = v_df.all().astype(int)
    # s_df[v_df.index] = v_df.values
    
    s_dfs.append(s_df)
stock_df = pd.concat(s_dfs, axis=0).reset_index(drop=True)

#Additional Features merge
stock_df = stock_df.merge(fg_df[['fish_group', 'fg:strain_name']].drop_duplicates(), how='left', on=['fish_group'])
 
#Light regime charts
# fig, ax = plt.subplots(1,4, figsize=(13,4), sharey=True)
# for i, name in enumerate(['Winter', 'Spring', 'Summer', 'Autumn']):
#     stock_df[f'light:{name}:cvg_perc'].plot.hist(ax=ax[i])
#     ax[i].set_title(f'{name}')
# fig.suptitle('FW Light regime cvg percs')
# plt.tight_layout()

#Stages charts
# fig, ax = plt.subplots(1,4, figsize=(13,4), sharey=True)
# for i, fw_prefix in enumerate(['I', 'H', 'F', 'OG/UPS']):
#     stock_df[f'cycle:len:{fw_prefix}'].plot.hist(ax=ax[i])
#     ax[i].set_title(f'Stage: {fw_prefix}')
# fig.suptitle('FW Stages lengths')
# plt.tight_layout()

#Modeling
import catboost as cb
import shap
from eli5.sklearn import PermutationImportance

#Dummies
data_df = stock_df[~stock_df['is_transfer']].reset_index(drop=True)
data_df = get_dummies(data_df, feature_name='fg:strain_name', prefix='fg:strain_name')
data_df = get_dummies(data_df, feature_name='transfer_season', prefix='calendar:transfer_season')
for prefix in ['I', 'H', 'F', 'OG/UPS']:
    data_df = get_dummies(data_df, feature_name=f'calendar:season:{prefix}', prefix=f'calendar:season:{prefix}')
    
#Features
feature_groups = ['vaccine:', 'cycle:', 'calendar:', 'fg:', 'temperature:', 'feed:', 'mortality:', 'light:']
feature_names = [col for col in data_df.columns for fg in feature_groups if fg in col and 'log' not in col]

#Train-test split  
               
TEST_PERC = 0.10
VAL_PERC = 0.15

#Target transform
log_eps = 1e-4
data_df['log_mortality:all'] = np.log(data_df['mrtperc_90d:all'] + log_eps)
data_df['log_mortality:env'] = np.log(data_df['mrtperc_90d:env'] + log_eps)

target_name = 'nSFR_90d:locus' #'mrtperc_90d:all' #_positive_rate
log_target_name = 'log_mortality:all'
log_transform = False

corr_vals = data_df[feature_names+[target_name,]].corr()
corr_vals = corr_vals.loc[target_name].to_frame()
corr_vals['abs'] = corr_vals.abs()
corr_vals = corr_vals.drop(target_name).sort_values('abs', ascending=True)

# corr_vals[target_name].plot(figsize=(8, 9), width=0.9, kind='barh', color=corr_vals[target_name].apply(lambda x: 'tab:blue' if x < 0 else 'tab:red'))
# plt.xlabel('Pearson correlation')
# plt.title(f'Simple feature groups to {target_name}')
# plt.tight_layout()

 
pi_dfs = []
for k in tqdm(range(0, 10), 'Permutation index. Random search'):
    data_df = split(data_df, random_state=k)
    val_idxs = data_df['input_type'] == 'val'
    data_df, model, metrics = train_evaluate_model(data_df, feature_names, target_name, loss_function='MAE')
    perm = PermutationImportance(model, cv="prefit").fit(
        data_df.loc[val_idxs, feature_names],
        data_df.loc[val_idxs,target_name]
    ) 
    pi_df = pd.DataFrame({ 'feature_name': feature_names,'pi': perm.feature_importances_,})
    pi_df['fold'] = k
    pi_dfs.append(pi_df)
pi_dfs = pd.concat(pi_dfs, axis=0)
# sns.boxplot(data=pi_dfs, x='pi', y='feature_name', showfliers=False)
pi_df = pi_dfs.groupby(['feature_name'])['pi'].mean().sort_values(ascending=False).reset_index()
pi_df = pi_df[pi_df['pi'] > 0]
feature_names = pi_df['feature_name'].tolist()

data_df = split(data_df, random_state=111)
data_df, model, metrics = train_evaluate_model(data_df, feature_names, target_name, verbose=1)

# s_df.append({'val_comb': val_comb, 'train_score': train_score, 'val_score': val_score})
# ax.set_xlim(0, 0.2)
# ax.set_ylim(0, 0.2)
x = np.linspace(data_df[target_name].min()-1e-3, data_df[target_name].max(), 20)
y = x
fig, ax = plt.subplots(1, 3, figsize=(12,4))
sns.scatterplot(data=data_df.loc[data_df['input_type']=='train'], x=target_name, y='prediction', color='tab:green', ax=ax[0], zorder=2)
ax[0].plot(x, y, ls='--', color='lightgrey', zorder=1)
ax[0].set_title('Train data')
sns.scatterplot(data=data_df.loc[data_df['input_type']=='val'], x=target_name, y='prediction', color='tab:orange', ax=ax[1], zorder=2)
ax[1].plot(x, y, ls='--', color='lightgrey', zorder=1)
ax[1].set_title('Val data')
sns.scatterplot(data=data_df.loc[data_df['input_type']=='test'], x=target_name, y='prediction', color='tab:red',ax=ax[2], zorder=2)
ax[2].plot(x, y, ls='--', color='lightgrey', zorder=1)
ax[2].set_title('Test data')

fig.suptitle(f'{target_name}; Actual to Prediction plot')
plt.tight_layout()
plt.show()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_df[feature_names])
explanation = shap.Explanation(values=shap_values, data=data_df, feature_names=feature_names)

#SHAP boxplots
shap_values = explainer(data_df[feature_names])
shap_i = pd.DataFrame({
    'feature': feature_names,
    'shap_importance':  np.abs(shap_values.values).mean(axis=0)
}).sort_values('shap_importance', ascending=False).reset_index(drop=True)

shap.summary_plot(shap_values, data_df[feature_names], plot_size=(12, 8))
# plt.xlim(-0.2,0.2)
plt.title(f'{target_name}. Baseline SHAP values')
plt.tight_layout()



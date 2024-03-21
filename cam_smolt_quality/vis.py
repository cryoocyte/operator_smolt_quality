import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cam_smolt_quality.configs import CURRENT_DATE

sns.scatterplot(data=data_df, x='cycle:transfer_weight', y='atpasa:latest_value', s=40, hue=f'{target_name}:bucket')
plt.title('Atpasa latest value vs transfer weight')
plt.tight_layout()

 'atpasa:latest_kfactor'
    
# 'light:summer:since_started'
##### Forecast cycle schedule
fig, ax = plt.subplots(2, figsize=(10,6))
stock_df['cycle:transfer_weight'].plot.hist(bins=40, ax=ax[0], color='tab:blue', label='weight')
for k, v in stock_df['cycle:transfer_weight'].quantile([0.25, 0.5, 0.75]).items():
    ax[0].axvline(v, ls='--', color='lightgrey')
ax[0].set_xlabel('Transfer weight')
stock_df['atpasa:latest_value'].plot.hist(bins=40, ax=ax[1], color='tab:red', label='atpasa')
for k, v in stock_df['atpasa:latest_value'].quantile([0.25, 0.5, 0.75]).items():
    ax[1].axvline(v, ls='--', color='lightgrey')
ax[1].set_xlabel('Atpasa latest weighted value')
# stock_df['light:summer:since_UPS'].plot.hist(bins=40, ax=ax[2], color='tab:green', label='light')
# ax[2].set_xlabel('Days from UPS until "Verano" light regime start')
ax[0].legend(); ax[1].legend(); #ax[2].legend()
fig.suptitle('Forecast time factors')
plt.tight_layout()

    
# fig, ax = plt.subplots(3, sharey=True)
# stock_df['light:days_in:Verano'].plot.hist(bins=50,ax=ax[0], color='tab:orange')
# ax[0].set_xlabel('Verano')
# stock_df['light:days_in:Invierno'].plot.hist(bins=50,ax=ax[1], color='tab:blue')
# ax[1].set_xlabel('Invierno')
# stock_df['light:days_in:Apagado'].plot.hist(bins=50,ax=ax[2], color='tab:grey')
# ax[2].set_xlabel('Apagado')
# fig.suptitle('Light regime in days')
# plt.tight_layout()
 

# fig, ax = plt.subplots(1, figsize=(13,4))
# sns.boxplot(data=stock_df, x='most_common_rsnperc', y='most_common_rsn',ax=ax, palette='tab20')
# fig.suptitle('AFTER DROP: Most common mortality reasons per fish group (first 90 days after transfer)')
# plt.tight_layout()

# plt.ioff()
# input_colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}
# data_df = data_df.sort_values('min_date')
# for g_id, g_df in data_df.groupby('site_name'):    
#     fig, ax = plt.subplots(figsize=(13, 6))
#     for i, row in g_df.iterrows():
#         plt.plot([row['min_date'], row['max_date']], [row['fish_group'], row['fish_group']], marker = 'o', color=input_colors[row['input_type']], linestyle = '-')
#     ax.set_title(f"SW Site name:{g_id}")
#     ax.set_xlim(pd.to_datetime(START_DATE,utc=True), pd.to_datetime(CURRENT_DATE,utc=True))
#     ax.set_ylabel('fish group')
#     ax.set_xlabel('event date')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'cam_smolt_quality/data/figures/site/sw/{g_id}.png')
#     plt.close()


# fig, ax = plt.subplots(1, 3, figsize=(12,4))
# data_df[target_name].plot.hist(bins=50, ax=ax[0])
# ax[0].set_title('Before log transformation')
# data_df.plot.scatter(target_name, log_target_name, ax=ax[1], zorder=2)
# ax[1].plot(
#     np.linspace(data_df[target_name].min(), data_df[target_name].max(), 5),
#     np.linspace(data_df[log_target_name].min(), data_df[log_target_name].max(), 5), ls='--', color='lightgrey', zorder=1)
# data_df[log_target_name].plot.hist(bins=50, ax=ax[2])
# ax[2].set_title('After log transformation')
# ax[1].set_title('Mortality target transformation technique')
# plt.tight_layout()

# data_df['transfer_month'] = data_df['transfer_month'].astype(str)



# target_name = 'log_mortality'
# fig, ax = plt.subplots(2, 3, figsize=(14,7), sharey=True)

# order = list(map(str, sorted(data_df['transfer_month'].astype(int).unique().tolist())))
# sns.stripplot(data=data_df, x='transfer_month', y=target_name, order=order, color='tab:green',ax=ax[0][0])
# means = data_df.groupby('transfer_month')[target_name].mean().loc[order]
# ax[0][0].plot(means.index, means.values,ls='--', color='lightgrey', zorder=0)
# ax[0][0].scatter(means.index, means.values, color='lightgrey', zorder=0)

# sns.scatterplot(data=data_df, x='cycle:fw_legnth', y=target_name, color='tab:blue', ax=ax[1][0])
# data_df['bins'] = pd.qcut(data_df['cycle:fw_legnth'], q=np.linspace(0, 1, 10))
# data_df['bins'] = data_df.groupby('bins')[target_name].transform('mean')
# r_df = data_df[['cycle:fw_legnth', 'bins']].sort_values('cycle:fw_legnth').drop_duplicates('bins', keep='last')
# sns.lineplot(data=r_df, x='cycle:fw_legnth', y='bins', color='lightgrey', ls='--', marker="o", ax=ax[1][0], zorder=0)

# sns.scatterplot(data=data_df, x='cycle:transfer_fish_weight', y=target_name, color='tab:orange', ax=ax[1][1])
# data_df['bins'] = pd.qcut(data_df['cycle:transfer_fish_weight'],q=np.linspace(0, 1, 10))
# data_df['bins'] = data_df.groupby('bins')[target_name].transform('mean')
# r_df = data_df[['cycle:transfer_fish_weight', 'bins']].sort_values('cycle:transfer_fish_weight').drop_duplicates('bins', keep='last')
# sns.lineplot(data=r_df, x='cycle:transfer_fish_weight', y='bins', color='lightgrey', marker="o", ls='--', ax=ax[1][1], zorder=0)

# order = data_df['fg:strain_name'].unique().tolist()
# sns.stripplot(data=data_df, x='fg:strain_name', y=target_name, order=order, color='tab:red', ax=ax[0][1])
# means = data_df.groupby('fg:strain_name')[target_name].mean().loc[order]
# ax[0][1].plot(means.index, means.values,ls='--', color='lightgrey', zorder=0)
# ax[0][1].scatter(means.index, means.values, color='lightgrey', zorder=0)

# sns.scatterplot(data=data_df, x='vaccine:days_since', y=target_name, color='tab:purple', ax=ax[0][2])
# data_df['bins'] = pd.qcut(data_df['vaccine:days_since'], q=np.linspace(0, 1, 10))
# data_df['bins'] = data_df.groupby('bins')[target_name].transform('mean')
# r_df = data_df[['vaccine:days_since', 'bins']].sort_values('vaccine:days_since').drop_duplicates('bins', keep='last')
# sns.lineplot(data=r_df, x='vaccine:days_since', y='bins', color='lightgrey', ls='--', marker="o", ax=ax[0][2], zorder=0)

# order = data_df['site_name'].unique().tolist()
# sns.stripplot(data=data_df, x='site_name', y=target_name, order=order, color='tab:brown',ax=ax[1][2])
# ax[1][2].tick_params(axis='x', rotation=90)
# # means = data_df.groupby('site_name')[target_name].mean().loc[order]
# # ax[0][2].plot(means.index, means.values,ls='--', color='lightgrey', zorder=0)
# # ax[0][2].scatter(means.index, means.values, color='lightgrey', zorder=0)


# data_df = get_dummies(data_df, feature_name='calendar:transfer_month', prefix='calendar:transfer_month')
# for prefix in ['H', 'F', 'OG', 'UPS']: #'I', 
    # data_df = get_dummies(data_df, feature_name=f'calendar:month:{prefix}', prefix=f'calendar:season:{prefix}')
    # data_df = utils.sincos_transform(data_df, f'calendar:dayofyear:{prefix}')
    # data_df = data_df.drop(f'calendar:dayofyear:{prefix}', axis=1)
# data_df = utils.sincos_transform(data_df, 'calendar:transfer_dayofyear')
# data_df = data_df.drop('calendar:transfer_dayofyear', axis=1)
    
# Cycle lengths
# data_df['transfer_len'].plot.hist(bins=30)
# plt.title('Transfer length in days at Site/Fish group level')
# plt.xlabel('Days')
# plt.tight_layout()

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


# colors = ['tab:blue' if x > 0 else 'tab:red' for x in dates_metrics_df['resid']]
# dates_metrics_df.plot.scatter('split_date', 'resid', color=colors, ax=ax[0][0], zorder=2)
# ax[0][0].axhline(0, ls='--', color='lightgrey')
# for index, row in dates_metrics_df.iterrows():
#     xx, yy = row['split_date'], row['resid']
#     ax[0][0].plot([xx, xx], [0, yy], color='lightgrey', linestyle='--', linewidth=0.95, zorder=1)
# # max_val = max(base_metrics_df['resid'].max(), abs(base_metrics_df['resid'].min()))
# # ax[0][0].set_ylim(-max_val, max_val)
# ax[0][0].set_title('T+1 test results based on split month')

# sns.barplot(data=month_metrics_df, x='month', y='mae', width=0.75, ax=ax[1][0], color='tab:purple')
# ax[0][1].set_title('Transfer month test errors')

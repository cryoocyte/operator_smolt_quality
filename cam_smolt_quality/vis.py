import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cam_smolt_quality.configs import CURRENT_DATE

# fig, ax = plt.subplots(1, figsize=(13,4))
# sns.boxplot(data=stock_df, x='most_common_rsnperc', y='most_common_rsn',ax=ax, palette='tab20')
# fig.suptitle('AFTER DROP: Most common mortality reasons per fish group (first 90 days after transfer)')
# plt.tight_layout()

# plt.ioff()
input_colors = {'train': 'tab:blue', 'val': 'tab:orange', 'test': 'tab:green'}
data_df = data_df.sort_values('min_date')
for g_id, g_df in data_df.groupby('site_name'):    
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, row in g_df.iterrows():
        plt.plot([row['min_date'], row['max_date']], [row['fish_group'], row['fish_group']], marker = 'o', color=input_colors[row['input_type']], linestyle = '-')
    ax.set_title(f"SW Site name:{g_id}")
    ax.set_xlim(pd.to_datetime(START_DATE,utc=True), pd.to_datetime(CURRENT_DATE,utc=True))
    ax.set_ylabel('fish group')
    ax.set_xlabel('event date')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'cam_smolt_quality/data/figures/site/sw/{g_id}.png')
    plt.close()


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




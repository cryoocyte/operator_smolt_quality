from sklearn.cluster import KMeans
import networkx as nx
from cam_smolt_quality.configs import CURRENT_DATE, PIPELINE_TYPE
import ecto_ds.procedural.utils as utils
import ecto_ds.procedural.math as math

from cam_smolt_quality.configs.cam_config import STRAIN_REPLACE_MAP, START_DATE

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

N_TRANSFER_DAYS = 90
OPERATION_MORTALITY_REASONS = ('Transporte', 'Lobo', 'Pajaro', 'Daño Mecánico', 'Daño Craneal', 'Daño Branquial', )


def draw_movememnts_tree(graph):
        
    def layer_position(G):
        # Initialize the layout with all node positions at (0, 0).
        sorted_nodes = sorted(G.nodes)
        pos = {node: [0, i] for i, node in enumerate(sorted_nodes)}
    
        # Set the x-position to the layer number, and y-position to the index within the sorted nodes.
        for layer in nx.topological_sort(G):
            successors = sorted(list(G.successors(layer)))
            for i, node in enumerate(successors):
                pos[node][0] = pos[layer][0] + 1
                pos[node][1] = sorted_nodes.index(node) + 1
        return pos
    
    plt.ion()
    subg = graph# nx.subgraph(graph, subgraph)
    pos = layer_position(subg)
    # pos = nx.spring_layout(subg)

    edge_labels = nx.get_edge_attributes(subg, 'scnt_f') 
    # Round scnt_f
    edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw(subg, pos, with_labels=True, arrows=True)
    nx.draw_networkx_edge_labels(subg, pos, edge_labels=edge_labels) 
    plt.show()
    
def get_fg_paths(movements_df, stock_df):
    m_df = movements_df.copy()
    m_df = m_df[m_df['src_fish_group'] != m_df['dst_fish_group']]    
    graph = nx.from_pandas_edgelist(
        m_df,
        source='src_fish_group',
        target='dst_fish_group',
        edge_attr=['transferred_cnt'],
        create_using=nx.DiGraph()
    )
    start_points = [node for node, degree in graph.in_degree() if degree == 0 or (degree == 1 and graph.has_edge(node, node))]
    # valid_end_points = [node for node, degree in graph.out_degree() if degree == 0 or (degree == 1 and graph.has_edge(node, node))]
    # valid_end_points = stock_df['fish_group'].unique().tolist()
    valid_end_points = stock_df['fish_group'].unique().tolist()
    
    cycle_id = 0
    fg_paths_df = []
    for dst_id in valid_end_points:
        # if dst_id == '56G2102.SNLCY2109.H.INV':
        #     break
        
        paths = [list(nx.all_simple_paths(graph, src_id, dst_id)) for src_id in start_points]
        paths = [p for pa in paths for p in pa if len(p) > 0]
        uniqs = list(set([p for path in paths for p in path]))
        if len(uniqs) > 0:
            fg_paths_df.append([uniqs, dst_id, cycle_id])
            cycle_id += 1
    fg_paths_df = pd.DataFrame(fg_paths_df, columns=['fish_group', 'dst_fish_group', 'cycle_id']).explode('fish_group')
    return fg_paths_df.drop_duplicates()

  
def eSFR(w, t):
    yf = (.2735797591)+(-.0720137809*t)+(.0187408253*t**2)+(-.0008145337*t**3)
    y0 = (-.79303459)+(.43059382*t)+(-.01471246*t**2)
    log_alpha = (-7.8284505676)+(.3748824960*t)+(-.0301640851*t**2)+(.0006516355*t**3)
    return (yf - (yf-y0)*np.exp(-np.exp(log_alpha)*w))

def get_raw_target(stock_df, mortality_df, sw_inv_df, sw_feed_df):
    
    keys = ['to_site_name', 'to_fish_group']
    stock_df = stock_df.groupby(keys).agg(
        min_transfer_date=('transfer_date', 'min'),
        max_transfer_date=('transfer_date', 'max'),
        stock_volume=('stock_volume', 'mean'),
        stock_cnt=('stock_cnt', 'sum'),
        stock_bms=('stock_bms', 'sum'),
    ).reset_index()
    stock_df['transfer_len'] = (stock_df['max_transfer_date'] - stock_df['min_transfer_date']).dt.days

    cycles_df = sw_inv_df.groupby(['fish_group', 'site_name', 'site_type']).agg(
        min_date=('event_date', 'min'),
        max_date=('event_date', 'max'),
        max_fish_wg=('fish_wg', lambda x: x.quantile(0.99))
    ).reset_index()
    cycles_df['days_length'] = (cycles_df['max_date'] - cycles_df['min_date']).dt.days.astype(int)
    cycles_df['is_finished'], cycles_df['is_transfer'] = True, False
    mask = (cycles_df['max_date'] >= pd.to_datetime(CURRENT_DATE, utc=True) - pd.Timedelta(days=7))
    cycles_df.loc[mask, 'is_finished'] = False
    cycles_df.loc[(mask & (cycles_df['days_length'] < N_TRANSFER_DAYS)), 'is_transfer'] = True
    # cycles_df = cycles_df[cycles_df['min_date'] != cycles_df['min_date'].min()] #Remove cycles that starts with MIN_DATE
    # cycles_df[cycles_df['fish_group']=='68G1702.SNFLY1712.H.PRIM']
    
    
    stock_df = stock_df.rename({'to_site_name': 'site_name', 'to_fish_group': 'fish_group'}, axis=1)
    keys = [ 'fish_group', 'site_name',]
    data_df = []
    i_dfs = []
    for tr_id, tr_df in tqdm(stock_df.groupby(keys), 'Get raw targets'):
        res_d = dict(zip(keys, tr_id))
        # if res_d['fish_group'] == '68G1702.SNFLY1712.H.PRIM':
        res_d['smolt_wg'] = tr_df['stock_bms'].item()/tr_df['stock_cnt'].item() * 1000
        res_d['stocking_density'] = tr_df['stock_bms'].item()/tr_df['stock_volume'].item() * 100 #g/m3
        
        first_day = tr_df['min_transfer_date'].item()
        last_transfer_day = tr_df['max_transfer_date'].item()
        last_day = first_day + pd.Timedelta(days=N_TRANSFER_DAYS)
        res_d['min_transfer_date'] = first_day
        res_d['last_transfer_date'] = last_transfer_day
        res_d['last_target_date'] = last_day

        mrt_df = mortality_df.copy()
        for key in keys: mrt_df = utils.base_filter(mrt_df, col=key, name=res_d[key])
        mrt_df = mrt_df.sort_values('event_date')
        mrt_df = mrt_df.loc[mrt_df['event_date'] <= last_day]

        i_df = sw_inv_df.copy()
        for key in keys: i_df = utils.base_filter(i_df, col=key, name=res_d[key])
        i_df = i_df.sort_values('event_date')
        i_df = i_df.loc[i_df['event_date'] <= last_day]
    
        f_df = sw_feed_df.copy()
        for key in keys: f_df = utils.base_filter(f_df, col=key, name=res_d[key])
        f_df = f_df.sort_values('event_date')
        f_df = f_df.loc[f_df['event_date'] <= last_day]
        i_df = pd.merge(i_df, f_df, how='left')
        
        #Mortality
        sum_mrts_all = mrt_df['mortality_cnt'].sum()
        mask = ~mortality_df['mortality_reason'].isin(OPERATION_MORTALITY_REASONS)         
        sum_mrts_env = mrt_df.loc[mask, 'mortality_cnt'].sum()
        
        res_d['mortality_cnt:all'] = sum_mrts_all
        res_d['mortality_cnt:env'] = sum_mrts_env
        res_d['stock_cnt'] = tr_df['stock_cnt'].item()
        res_d['stock_bms'] = tr_df['stock_bms'].item()

        mrt_value_all = sum_mrts_all/tr_df['stock_cnt'].item() * 100
        mrt_value_env = sum_mrts_env/tr_df['stock_cnt'].item() * 100
        res_d[f'mrtperc_{N_TRANSFER_DAYS}d:all'] = mrt_value_all
        res_d[f'mrtperc_{N_TRANSFER_DAYS}d:env'] = mrt_value_env

        if len(mrt_df) > 0:
            by_rsn = mrt_df.groupby('mortality_reason')['mortality_cnt'].sum()
            by_rsn = by_rsn/by_rsn.sum()
            mc_rsn = by_rsn.idxmax() #Most common reason
            mc_perc = by_rsn[mc_rsn]
            res_d['most_common_rsn'] = mc_rsn
            res_d['most_common_rsnperc'] = mc_perc
   
        #TGC
        #Avoid absurd weight drops
        rolling_max = i_df['fish_wg'].rolling(7).max()
        min_wg = rolling_max.min()
        if len(i_df) >= N_TRANSFER_DAYS:
            max_wg = rolling_max.loc[(i_df['event_date']-last_day).dt.days.abs().idxmin()].item()
            # log = (np.log(max_wg) - np.log(min_wg))
            # sgr_value = log/N_TRANSFER_DAYS * 100
        else: 
            max_wg = rolling_max.max()
            # log = (np.log(max_wg) - np.log(min_wg))
            # sgr_value = log/len(i_df) * 100
        tgc_value = (max_wg**(1/3)-min_wg**(1/3))/i_df['degree_days'].sum() * 100
        # res_d[f'SGR_{N_TRANSFER_DAYS}d'] = sgr_value
        res_d[f'TGC_{N_TRANSFER_DAYS}d'] = tgc_value

        #nSFR
        osfr_value = i_df['feed_amount']/i_df['start_fish_bms'] * 100
        esfr_value = eSFR(i_df['fish_wg'], i_df['degree_days'])
        nsfr_values = (osfr_value/esfr_value)
        avg_nsfr_value = nsfr_values.mean()
        res_d[f'nSFR_{N_TRANSFER_DAYS}d'] = avg_nsfr_value
        
        # i_df['nsfr_values'] = nsfr_values
        # res_d[f'nSFR_positive_rate_{N_TRANSFER_DAYS}d'] = (i_df.groupby('locus_id')['nsfr_values'].mean() >= 1.0).mean()
        res_d['number_of_cages'] = i_df['locus_id'].nunique()

        data_df.append(res_d)
    stock_df = pd.DataFrame(data_df)
    stock_df = pd.merge(stock_df, cycles_df, on=['site_name', 'fish_group'], how='left')

    return stock_df

def target_full_analysis(stock_df, use_all=True, draw=True):
    
    is_finished = stock_df['is_finished']==True
    model = KMeans(n_clusters=3, random_state=42)
    x = stock_df.loc[is_finished, 'days_length'].values[:,np.newaxis]
    model.fit(x)
    stock_df.loc[is_finished, 'cycle_cluster'] = model.predict(x).astype(str)

    # fig, ax = plt.subplots()
    # colors = dict(zip(['0', '1', '2'], ['tab:orange', 'tab:blue', 'tab:green']))
    # sns.boxplot(data=stock_df[(is_finished & (stock_df['days_length'] > 90))], x='nSFR_90d', y='cycle_cluster',palette=colors,order=['0', '1', '2'],ax=ax)


    if use_all:
        valid_stocks_df = stock_df[(~is_finished | (stock_df['days_length'] > N_TRANSFER_DAYS))].reset_index(drop=True)
    else:
        bg = stock_df.groupby('cycle_cluster')['days_length'].min().idxmax()
        valid_stocks_df = stock_df[((stock_df['cycle_cluster']==bg) | ~is_finished)].reset_index(drop=True)
    valid_no_transfer = valid_stocks_df['is_transfer']==False
    valid_stocks_df['transfer_year'] = valid_stocks_df['min_date'].dt.year.astype(str)
               
    if draw:
        fig, ax = plt.subplots(2, 5, figsize=(16,8))
        sorted_years = sorted(valid_stocks_df['transfer_year'].unique())
        years_palette = dict(zip(sorted_years, sns.color_palette("tab20", len(sorted_years))))
        colors = dict(zip(['0', '1', '2'], ['tab:orange', 'tab:blue', 'tab:green']))
        for g_id, g_df in stock_df.loc[is_finished].groupby('cycle_cluster'):
            g_df['days_length'].plot.hist(bins=15, label=f'Cluster: {g_id}', color=colors[g_id], ax=ax[0][0])
        ax[0][0].legend()
        ax[0][0].set_xlabel('Cycle length')
        ax[0][0].set_title(f'CAM: Distribution of all SW cycle lengths\nsince {stock_df["min_date"].min().year}')
        
        r_df = valid_stocks_df[valid_no_transfer].groupby('transfer_year').size().rename('cnt').reset_index()
        sns.barplot(data=r_df, x='cnt', y='transfer_year', order=sorted_years,palette=years_palette, ax=ax[1][0])
        ax[1][0].set_xlabel('count')
        ax[1][0].set_title(f'Cycles count. CLuster: {bg} \n(Using year class)')
        
        #Mortality: all
        valid_stocks_df.loc[valid_no_transfer, 'mrtperc_90d:all'].plot.hist(bins=50, ax=ax[0][1], color='tab:purple' if use_all else colors[bg], label=None if use_all else f'Cluster: {bg}')
        ax[0][1].set_xlabel('%')
        ax[0][1].set_xlim(0, 10)
        ax[0][1].set_title('Mortality percentage: All')
        ax[0][1].legend()
        sns.stripplot(
            data=valid_stocks_df[valid_no_transfer],
            y='transfer_year', x='mrtperc_90d:all'
            , hue='transfer_year', palette=years_palette,
            order=sorted_years, ax=ax[1][1],  legend=None
        )
        ax[1][1].set_title('Mortality percentage: All.\n(Using year class)')
        ax[1][1].set_xlabel('%')
        ax[1][1].set_xlim(0, 10)
        
        #Mortality: env
        valid_stocks_df.loc[valid_no_transfer, 'mrtperc_90d:env'].plot.hist(
            bins=50, ax=ax[0][2], color='tab:purple' if use_all else colors[bg], label=None if use_all else f'Cluster: {bg}')
        ax[0][2].set_xlabel('%')
        ax[0][2].set_xlim(0, 10)
        ax[0][2].set_title('Mortality percentage: Env only')
        ax[0][2].legend()
        sns.stripplot(
            data=valid_stocks_df[valid_no_transfer],
            y='transfer_year', x='mrtperc_90d:env'
            , hue='transfer_year', palette=years_palette,
            order=sorted_years, ax=ax[1][2],  legend=None
        )
        ax[1][2].set_title('Mortality percentage: Env only.\n (Using year class)')
        ax[1][2].set_xlabel('%')
        ax[1][2].set_xlim(0, 10)
        
        #nSFR positive rate
        valid_stocks_df.loc[valid_no_transfer, 'nSFR_positive_rate_90d'].plot.hist(
            bins=50, ax=ax[0][3], color='tab:purple' if use_all else colors[bg], label=None if use_all else f'Cluster: {bg}')
        ax[0][3].set_xlabel('%')
        ax[0][3].set_xlim(0.0, 100)
        ax[0][3].set_title('nSFR rate')
        ax[0][3].legend()
        sns.stripplot(
            data=valid_stocks_df[valid_no_transfer],
            y='transfer_year', x='nSFR_positive_rate_90d'
            , hue='transfer_year', palette=years_palette,
            order=sorted_years, ax=ax[1][3],  legend=None
        )
        ax[1][3].set_title('nSFR Positive rate.\n(Using year class)')
        ax[1][3].set_xlabel('%')
        ax[1][3].set_xlim(0.0, 100)
        
        #nSFR
        valid_stocks_df.loc[valid_no_transfer, 'nSFR_90d'].plot.hist(
            bins=40, ax=ax[0][4], color='tab:purple' if use_all else colors[bg], label=None if use_all else f'Cluster: {bg}')
        ax[0][4].set_xlabel('%')
        ax[0][4].set_xlim(0.2, 1.8)
        ax[0][4].set_title('nSFR')
        ax[0][4].legend()
        sns.stripplot(
            data=valid_stocks_df[valid_no_transfer],
            y='transfer_year', x='nSFR_90d'
            , hue='transfer_year', palette=years_palette,
            order=sorted_years, ax=ax[1][4],  legend=None
        )
        ax[1][4].set_title('nSFR (locus level).\n(Using year class)')
        ax[1][4].set_xlabel('%')
        ax[1][4].set_xlim(0.2, 1.8)
        
        fig.suptitle('CAM/SW/Site level/ First 90days summary. Finished transfers')
        plt.tight_layout()
        
     
    valid_stocks_df['transfer_month'] = valid_stocks_df['min_date'].dt.month#.astype(str)
    valid_stocks_df['transfer_season'] = valid_stocks_df['transfer_month'].apply(utils.season_from_month)
    return valid_stocks_df

def prep_fishgroups(fg_df):
    #Fish group prep
    fg_df['strain_name'] = fg_df['strain_name'].replace(STRAIN_REPLACE_MAP)
    fg_df = utils.add_prefix(fg_df, keys=['fish_group', 'fish_group_id'], prefix_name='fg')
    return fg_df

def prep_vaccines(vaccines_df):
    #Vaccine prep
    vaccines_df['type'] = vaccines_df['type'].str.lstrip("0")
    vaccines_df['type'] = vaccines_df['type'].apply(lambda x: x.split('/')[0])
    vaccines_df['type'] = vaccines_df['type'].replace({'SRSv': 'SRS'})
    type_names = ['IPN', 'SRS', 'VVV', 'AAA', 'ISA', 'VIB', 'BKD']
    for vtype in type_names:
        vaccines_df[f'prefix:{vtype}'] = vaccines_df['type'].apply(lambda x: True if vtype in x else False)
    vaccines_df = utils.add_prefix(vaccines_df, keys=['event_date', 'site_name', 'fish_group'], prefix_name='vaccine')
    return vaccines_df

#Sensors
def join_logbook_jobs(logbook_df, jobs_df):
    sensors_df = pd.concat([logbook_df, jobs_df], axis=0)
    sensors_df = sensors_df.drop_duplicates(keep='last') #Keep jobs data
    sensors_df = sensors_df[sensors_df['variable'].notna()]
    sensors_df = sensors_df.pivot_table(values='value', columns=['variable'], index=['event_date', 'locus_group_id'], aggfunc='mean')
    sensors_df = utils.add_prefix(sensors_df, keys='', prefix_name='sensor').reset_index()
    return sensors_df


def get_feed(feed_df):
    feed_df['has_luf'] = feed_df['feed_name'].apply(lambda x: True if ' LUF' in x else False)
    return feed_df


def get_jobs(jobs_df, fw_inv_df, locus_locus_group_df, sqs=[0.05, 0.25, 0.5, 0.75, 0.95]):
    jobs_cols = [c for c in jobs_df.columns if 'sensor' in c] #['sensor:Temperature',] #TODO!
    sn_df = fw_inv_df.copy()
    sn_df = sn_df.merge(locus_locus_group_df[['locus_id', 'locus_group_id']], on='locus_id')
    sn_df = sn_df.merge(jobs_df, on=['event_date', 'locus_group_id'], how='left')
    sn_df = sn_df[~sn_df[jobs_cols].isna().all(1)]
    sn_df = sn_df.groupby(['event_date', 'fish_group', 'fw_locus_prefix'])[jobs_cols].mean().reset_index()
    sn_df = sn_df[~sn_df[jobs_cols].isna().all(1)]
    q_df = []
    for col in jobs_cols:
        lw, uw = math.get_outliers(sn_df, col)
        sn_df.loc[((sn_df[col] >= uw) | (sn_df[col] <= lw)), col] = np.nan
        res = sn_df.groupby('fw_locus_prefix')[col].quantile(sqs).reset_index()
        res = res.pivot_table(index='level_1', columns='fw_locus_prefix', values=col, aggfunc='mean')
        res['feature_name'] = col
        q_df.append(res)
    q_df = pd.concat(q_df, axis=0)
    return sn_df, q_df

def get_weighted_date(df, w):
    first_date = pd.to_datetime('1970-01-01', utc=True)
    int_date = (df['event_date'] - first_date).dt.days
    days = int((int_date * df[w]/df[w].sum()).sum())
    first_date = first_date + pd.Timedelta(days=days)
    return first_date

def construct_dataset(
        stock_df, fg_paths_df,fg_df, fw_inv_df,
        sw_inv_df, vaccines_df, fw_feed_df,
        fw_mortality_df, treatments_df, atpasa_df,
        jobs_df, jobs_q_df, luf_fish_df, sensors_df
    ):
    s_dfs = []
    sns_df = []
    for (uniq_id, dst_fish_group), uniq_df in tqdm(fg_paths_df.groupby(['cycle_id', 'dst_fish_group'])):
        # if dst_fish_group == '31G1902.SNFLY1904.M.OTO':
        # pass
        '''
        sensors extraction, well done
        fish_luf data
        
        
        
        '''
        
        #Feature Groups
        uniq_fgs = uniq_df['fish_group'].tolist()
        # uniq_fgs.remove(dst_fish_group)
        fw_df = fw_inv_df.loc[fw_inv_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        fw_df = fw_df.sort_values('event_date')
        sw_df = sw_inv_df.loc[sw_inv_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        s_df = stock_df[stock_df['fish_group']==dst_fish_group].reset_index(drop=True)
        v_df = vaccines_df.loc[vaccines_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        fe_df = fw_feed_df.loc[fw_feed_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        mrt_df = fw_mortality_df.loc[fw_mortality_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        trt_df = treatments_df.loc[treatments_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        trt_df = trt_df.groupby(['event_date', 'active_substance_name'])['amount'].sum().reset_index()
        at_df = atpasa_df.loc[atpasa_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        jb_df = jobs_df.loc[jobs_df['fish_group'].isin(uniq_fgs)].reset_index(drop=True)
        sn_df = sensors_df[((sensors_df['event_date']>=fw_df['event_date'].min()) & (sensors_df['event_date']<=fw_df['event_date'].max()))]
        lf_df = luf_fish_df.loc[luf_fish_df['locus_id'].isin(fw_df['locus_id'].unique()) & ((luf_fish_df['event_date']>=fw_df['event_date'].min()) & (luf_fish_df['event_date']<=fw_df['event_date'].max()))].reset_index(drop=True)

        if 'pH Sensor - UPS 1' in sn_df['sensor_name'].unique().tolist():
            break
        
        keys = ['event_date', 'site_name', 'fish_group', 'fw_locus_prefix']
        i_df = fw_df.groupby(keys)['fish_wg'].mean().reset_index()
        sn_df = i_df.merge(sn_df).pivot_table(index=keys, columns=['sensor_type_name'], values='value',aggfunc='mean').reset_index()
        
        
        
        #Weighted stages length
        weights_df = pd.pivot_table(data=fw_df, index='fish_group', columns='fw_locus_prefix', values='end_fish_bms', aggfunc='max')
        weights_df = weights_df.divide(weights_df.sum(0), axis=1)
        
        # Smolt size and weight
        s_df = s_df.rename({'smolt_wg': 'cycle:transfer_weight'}, axis=1)
        
        # Smolt condition factor (K) \ Atpasa
        #Atpasa
        if len(at_df) > 0:
            res = []
            for g_id, g_df in at_df.groupby('fish_group'):
               g_df = g_df[g_df['event_date']==g_df['event_date'].max()]
               weights = g_df['n_samples']/g_df['n_samples'].sum()
               res.append({
                   'fish_group': g_id,
                   'event_date': g_df['event_date'].max(),
                   'atpasa': (g_df['atpasa'] * weights).sum(),
                   'kfactor': (g_df['k_factor'] * weights).sum()
               })
            res = pd.DataFrame(res).set_index('fish_group')
            s_df['atpasa:latest_date'] = (weights_df.loc[res.index, 'UPS'] * res['atpasa']).sum()
            s_df['atpasa:latest_kfactor'] = (weights_df.loc[res.index, 'UPS'] * res['kfactor']).sum()
            # at_df['event_week'] = at_df['event_date'].dt.strftime('%Y-%m-%w')
            # l_at = at_df.groupby('event_date').apply(lambda x: (x['n_samples'] * x['atpasa']).sum() / x['n_samples'].sum()).sort_index()
            # l_at = l_at.diff()
            # l_at[l_at < 0] = 0
            # s_df['atpasa:derivative'] = l_at.diff().mean().item()
     
        # Stocking density
        s_df = s_df.rename({'stocking_density': 'cycle:stocking_density'}, axis=1)

        # Lufenuron container (lab)
    
             
             
                    
        if sw_df['event_date'].min() <= pd.to_datetime(START_DATE, utc=True) or len(fw_df) == 0:
            # s_dfs.append(s_df)
            continue
        
        uniq_stages = fw_df['fw_locus_prefix'].unique()
        fw_stages_sorted = ['I', 'H', 'F', 'OG', 'UPS']
        if any(stage not in uniq_stages for stage in fw_stages_sorted) and PIPELINE_TYPE != 'forecast':
            continue
        fw_df = fw_df[fw_df['fw_locus_prefix'].isin(fw_stages_sorted)].reset_index(drop=True)
        
        #First dates
        # first_feeding_date = fe_df['event_date'].min() #Simple, TODO!!!
        first_vaccine_date = get_weighted_date(v_df, 'vaccine:fish_cnt') #Weighted
        
        # sns.scatterplot(data=fw_df[fw_df['fw_locus_prefix'] == 'I'], x='event_date', y='fish_wg', hue='fish_group')
        # sns.scatterplot(data=fw_df, x='event_date', y='fish_wg', hue='fish_group')
        # sns.scatterplot(data=fw_df, x='event_date', y='fish_wg', hue='fw_locus_prefix')
        
        #Cycle features
        s_df['calendar:transfer_dayofyear'] = s_df['min_transfer_date'].dt.dayofyear.item()
        
        ## FW stages length
        #Fix incubation issues
        # fw_df = fw_df.loc[fw_df.drop(['event_date'], axis=1).drop_duplicates().index]
        # i_mask = fw_df['fw_locus_prefix']=='I'
        # if fw_df.loc[i_mask, 'fish_group'].nunique() == 1:
        #     h_min = fw_df.loc[fw_df['fw_locus_prefix']=='H', 'event_date'].min()
        #     fw_df.loc[(i_mask & (fw_df['event_date'] > h_min)), 'fw_locus_prefix'] = np.nan

        cnts_df = pd.pivot_table(data=fw_df, index='fish_group', columns='fw_locus_prefix', values='event_date', aggfunc='nunique')
        cnts_df[cnts_df == 0] = np.nan
        fw_stages_lengths = (cnts_df * weights_df).sum(0).astype(int)
            
        for k in fw_stages_lengths.index:
            if fw_stages_lengths[k] == 0: res = np.nan
            else: res = fw_stages_lengths[k]
            s_df[f'cycle:len:{k}'] = res
    
    
        #Weighted stages day
        dates_df = pd.pivot_table(data=fw_df[(fw_df['fw_locus_prefix']!='I') & (fw_df['fw_locus_prefix'].isin(fw_stages_sorted))], index='fish_group', columns='fw_locus_prefix', values='event_date', aggfunc='min')
        
        for s in dates_df.columns: dates_df[s] = dates_df[s].dt.dayofyear
        dates = (dates_df * weights_df.loc[dates_df.index, dates_df.columns]).sum(0).astype(int)
        dates.index = [f'calendar:dayofyear:{i}' for i in dates.index]    
        dates = dates.to_frame().T
        dates.index = s_df.index
        s_df = pd.concat([s_df, dates], axis=1)
        
        #Light regime
        # fw_df = pd.merge(fw_df, fw_light_df, how='left', on=['event_date', 'site_name', 'locus_id', 'fw_locus_prefix'])
        # fw_df['light_regime'] = fw_df.groupby('locus_id')['light_regime'].ffill()
        # l_df = fw_df.loc[((fw_df['light_regime'] != 'Apagado') & (fw_df['light_regime'].notna()))].reset_index(drop=True)
        # # l_df = fw_df.loc[fw_df['light_regime'] == 'Verano'].reset_index(drop=True)
        # l_df['season'] = l_df['event_date'].dt.month.apply(utils.season_from_month)
        # l_df = l_df[l_df['season'] == 'Q3'].reset_index(drop=True) #Summer
        # l_df['event_week'] = l_df['event_date'].dt.strftime('%Y-%m-%w')
        # n_of_weeks = l_df['event_week'].nunique()
        # s_df['light:summer_weeks'] = n_of_weeks
        
        #Sensor
        # sns.scatterplot(data=sn_df, x='event_date', y='sensor:Temperature', hue='fish_group')
        sensor_cols = ['sensor:Temperature',] #[c for c in sensors_df.columns if 'sensor' in c] # #TODO!
    
        for sensor_col in sensor_cols:
            for aggfunc in ['mean', 'min', 'max', 'std']:
                sg_df = pd.pivot_table(data=sn_df, index='fish_group', columns='fw_locus_prefix', values=sensor_col, aggfunc=aggfunc)
                res = (sg_df * weights_df).sum(0)
                res[res==0] = np.nan
                res.index = [f'{sensor_col}:{aggfunc}:{i}' for i in res.index]    
                for k, v in res.items():
                    s_df[k] = v
            sq_df = sensors_q_df[sensors_q_df['feature_name']==sensor_col]
            for q, vals in sq_df.iterrows():
                for prefix, g_df in sn_df.groupby('fw_locus_prefix'):
                    perc = (g_df[sensor_col] <= vals[prefix]).mean() * 100
                    s_df[f'{sensor_col}:perc_q<={q}:{prefix}'] = perc
           
                    
        #Temperature last N weeks
        tmp_df = sn_df[['event_date', 'sensor:Temperature']]
        for N in [1,2,3,4]:
            last_date = s_df['min_transfer_date'].item()-pd.Timedelta(weeks=N)
            t_df = tmp_df.loc[tmp_df['event_date'] >= last_date, 'sensor:Temperature']
            s_df[f'sensor:Temperature:mean:last_{N}w'] = t_df.mean()
            
    
        #Vaccines    
        #Define vaccine date/since days based on weighted fish cnt strategy
        days_since_vaccine = (s_df['min_transfer_date'].item() - first_vaccine_date).days
        s_df['vaccine:days_since_started'] = days_since_vaccine
        s_df['vaccine:days_of'] = v_df['event_date'].nunique()
        #Vaccine weight
        cnts_df = pd.merge(v_df.groupby('event_date')['vaccine:fish_cnt'].sum().reset_index(), fw_df.groupby('event_date')['fish_wg'].mean().reset_index())
        fish_wg = sum(cnts_df['fish_wg'] * cnts_df['vaccine:fish_cnt']/cnts_df['vaccine:fish_cnt'].sum())
        s_df['vaccine:mean_fish_wg'] = fish_wg
        
        #Overall oSFR
    
        #nSFR OG, UPS
        g_df = fw_df[fw_df['fw_locus_prefix'].isin(['OG', 'UPS'])]
        bms_df = g_df.groupby(['event_date', 'fish_group', 'fw_locus_prefix']).agg(
            start_fish_bms=('start_fish_bms', 'sum'),
            degree_days=('degree_days', 'mean'),
            fish_wg=('fish_wg', 'mean')
        ).reset_index()
        bms_df = pd.merge(bms_df, fe_df, how='left', on=['event_date', 'fish_group'])

        # bms = g_df['end_fish_bms'].max() # - g_df['end_fish_bms'].min()
        # min_date, max_date = g_df['event_date'].min(), g_df['event_date'].max()
        # feed = fe_df.loc[((fe_df['event_date'] >= min_date) & (fe_df['event_date'] <= max_date)), 'feed_amount']
        bms_df['osfr_value'] = bms_df['feed_amount']/bms_df['start_fish_bms'] * 100
        bms_df['esfr_value'] = eSFR(bms_df['fish_wg'], bms_df['degree_days'])#.mean()
        bms_df['nsfr_value'] = (bms_df['osfr_value']/bms_df['esfr_value'])
        nsfr_df = pd.pivot_table(data=bms_df, index='fish_group', columns='fw_locus_prefix', values='nsfr_value', aggfunc='mean')
        nsfr_values = (nsfr_df * weights_df.loc[nsfr_df.index, nsfr_df.columns]).sum(0)
        nsfr_values.index = [f'feed:nSFR:{i}' for i in nsfr_values.index]    
        nsfr_values = nsfr_values.to_frame().T
        nsfr_values.index = s_df.index
        s_df = pd.concat([s_df, nsfr_values], axis=1)

        for N in [1,2,3,4]:
            last_date = s_df['min_transfer_date'].item()-pd.Timedelta(weeks=N)
            t_df = bms_df.loc[bms_df['event_date'] >= last_date]
            s_df[f'feed:oSFR:mean:last_{N}w'] = t_df['osfr_value'].mean()
            s_df[f'feed:nSFR:mean:last_{N}w'] = t_df['nsfr_value'].mean()
    
        #Feed LUF
        luf_perc = fe_df.loc[fe_df['has_luf']==True, 'feed_amount'].sum()/fe_df['feed_amount'].sum() * 100
        s_df['feed:luf_perc'] = luf_perc
        
        #TGC: F, OG/UPS
        for g_id, g_df in fw_df.groupby('fw_locus_prefix'):
            if g_id not in ['I', 'H'] and g_id in fw_stages_sorted:
                tgc_value = (g_df.groupby('fish_group')['fish_wg'].max() - g_df.groupby('fish_group')['fish_wg'].min())/g_df.groupby('fish_group')['degree_days'].sum() * 100
                tgc_value = (tgc_value * weights_df.loc[weights_df[g_id].notna(), g_id]).sum()
                s_df[f'cycle:TGC:{g_id}'] = tgc_value
        
        #Mortality rate since Fry stage
        m_df = pd.pivot_table(data=mrt_df[~mrt_df['fw_locus_prefix'].isin(['I', 'H']) & (mrt_df['fw_locus_prefix'].isin(fw_stages_sorted))], index='fish_group', columns='fw_locus_prefix', values='mortality_cnt', aggfunc='sum')
        rates = (m_df * weights_df.loc[m_df.index, m_df.columns]).sum(0)
        mrt_rate = rates/(s_df['stock_cnt'].item()+rates)
        mrt_rate.index = [f'mortality:mrt_rate:{i}' for i in mrt_rate.index]    
        mrt_rate = mrt_rate.to_frame().T
        mrt_rate.index = s_df.index
        s_df = pd.concat([s_df, mrt_rate], axis=1)
        for N in [1,2,3,4]:
            last_date = s_df['min_transfer_date'].item()-pd.Timedelta(weeks=N)
            mrts = mrt_df.loc[mrt_df['event_date'] >= last_date, 'mortality_cnt'].sum()
            s_df[f'mortality:mrt_rate:last_{N}w'] = mrts/(s_df['stock_cnt'].item()+mrts)
    
        
        #Treatments
        days = trt_df.groupby('active_substance_name')['event_date'].nunique()
        amount = trt_df.groupby('active_substance_name')['amount'].sum()
        amount_per_day = amount.divide(days)
        for k, v in amount_per_day.items():
            s_df[f'treatment:amount_per_day:{k}'] = v
        
        s_df = s_df.copy()
        for g_id, g_df in trt_df.groupby('active_substance_name'):
            g_df = g_df.sort_values('event_date')
            days_in_treatment = g_df['event_date'].nunique()
            max_consectuive = g_df['event_date'].diff().dt.days.max()
            last_treatment = (fw_df['event_date'].max() - g_df['event_date'].max()).days
            s_df.loc[s_df.index, f'treatment:days_in:{g_id}'] = days_in_treatment
            s_df.loc[s_df.index,f'treatment:consecutive_without:{g_id}'] = max_consectuive
            s_df.loc[s_df.index,f'treatment:days_since_last:{g_id}'] = last_treatment
        s_dfs.append(s_df)
    
    stock_df = pd.concat(s_dfs, axis=0).reset_index(drop=True)
    
    #Additional Features merge
    stock_df = stock_df.merge(fg_df[['fish_group', 'fg:strain_name']].drop_duplicates(), how='left', on=['fish_group'])
    return stock_df


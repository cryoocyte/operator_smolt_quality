import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
from cam_smolt_quality.configs import READ_PARAMS
from tqdm import tqdm

QUERIES_FILEPATH = 'cam_smolt_quality/queries/'
CSV_BUFFER_FILEPATH = 'cam_smolt_quality/data/'


def read_file(filename, conn_params=None, buffer=False, save=True):
    print(f'Reading {filename}, from {"csv" if buffer else "sql"}')
    query_path = os.path.join(QUERIES_FILEPATH, filename+'.sql')
    csv_path = os.path.join(CSV_BUFFER_FILEPATH, filename+'.csv')

    if not buffer:
        with open(query_path, 'r') as file:
            query = file.read()
        connection_string = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        with create_engine(connection_string).connect() as engine:
            df = pd.read_sql(query, con=engine) 
        if save:
            print(f'-- Saving {filename}')
            df.to_csv(csv_path, index=False)   
    else:
        df = pd.read_csv(csv_path)
    date_cols = [col for col in df.columns if 'date' in col or 'time' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df

def eSFR (row):
    w = row['open_weight']
    t = row['degree_days']
    yf = (.2735797591)+(-.0720137809*t)+(.0187408253*t**2)+(-.0008145337*t**3)
    y0 = (-.79303459)+(.43059382*t)+(-.01471246*t**2)
    log_alpha = (-7.8284505676)+(.3748824960*t)+(-.0301640851*t**2)+(.0006516355*t**3)
    return (yf - (yf-y0)*np.exp(-np.exp(log_alpha)*w))

def generate_event_dates(row):
    event_dates = pd.date_range(row['transfer_date'], row['sw90_date'], freq='D')
    return pd.DataFrame({'locus_id': row['locus_id'],
                         'fish_group_id': row['fish_group_id'],
                         'transfer_year': row['transfer_year'],
                         'event_date': event_dates})

def weighted_avg(x, weight, factor):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tmp = x[[weight, factor]].dropna()
        weighted_sum = (tmp[weight] * tmp[factor]).sum()
        count_sum = tmp[weight].sum()
        return weighted_sum / count_sum

def expand_dates_vectorized(df):
    df = df.loc[df.index.repeat((df.endtime - df.starttime).dt.days)]
    df['event_date'] = df.groupby(level=0)['starttime'].transform(lambda x: x + pd.to_timedelta(np.arange(len(x)), 'D'))
    df = df.drop(columns=['starttime', 'endtime'])
    df = df.rename({'count_ratio': 'weight0'}, axis=1)
    return df


def read_file(filename, buffer=False, save=True, encoding='utf-8', sep=','):
    print(f'Reading {filename}, from {"csv" if buffer else "sql"}')
    query_path = os.path.join(QUERIES_FILEPATH, filename+'.sql')
    csv_path = os.path.join(CSV_BUFFER_FILEPATH, filename+'.csv')
    
    conn_params = READ_PARAMS
    if not buffer:
    
        with open(query_path, 'r', encoding=encoding) as file:
            query = file.read()
        connection_string = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        with create_engine(connection_string).connect() as engine:
            df = pd.read_sql(query, con=engine) 
        if save:
            print(f'-- Saving {filename}')
            df.to_csv(csv_path, index=False)   
    else:
        df = pd.read_csv(csv_path,encoding=encoding, sep=sep)
    date_cols = [col for col in df.columns if 'date' in col or 'time' in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], utc=True)
    return df

def eSFR (row):
    w = row['open_weight']
    t = row['degree_days']
    yf = (.2735797591)+(-.0720137809*t)+(.0187408253*t**2)+(-.0008145337*t**3)
    y0 = (-.79303459)+(.43059382*t)+(-.01471246*t**2)
    log_alpha = (-7.8284505676)+(.3748824960*t)+(-.0301640851*t**2)+(.0006516355*t**3)
    return (yf - (yf-y0)*np.exp(-np.exp(log_alpha)*w))

def generate_event_dates(row):
    event_dates = pd.date_range(row['transfer_date'], row['sw90_date'], freq='D')
    return pd.DataFrame({'locus_id': row['locus_id'],
                         'fish_group_id': row['fish_group_id'],
                         'transfer_year': row['transfer_year'],
                         'event_date': event_dates})

def weighted_avg(x, weight, factor):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tmp = x[[weight, factor]].dropna()
        weighted_sum = (tmp[weight] * tmp[factor]).sum()
        count_sum = tmp[weight].sum()
        return weighted_sum / count_sum

def expand_dates_vectorized(df):
    df = df.loc[df.index.repeat((df.endtime - df.starttime).dt.days)]
    df['event_date'] = df.groupby(level=0)['starttime'].transform(lambda x: x + pd.to_timedelta(np.arange(len(x)), 'D'))
    df = df.drop(columns=['starttime', 'endtime'])
    df = df.rename({'count_ratio': 'weight0'}, axis=1)
    return df


class FeedFactorsCalculator:
    """Calculates numeric feed factors such as FCR, SGR, SFR"""

    @staticmethod
    def calculate_fcr(sum_feed: np.ndarray,
                      fish_count_end: np.ndarray,
                      average_weight_end: np.ndarray,
                      fish_count_start: np.ndarray,
                      average_weight_start: np.ndarray):
        return (1000 * sum_feed) / (fish_count_end * average_weight_end - fish_count_start * average_weight_start)

    @staticmethod
    def calculate_sgr(average_weight_end: np.ndarray, average_weight_start: np.ndarray, period_len: np.ndarray):
        return 100 * ((average_weight_end / average_weight_start) ** (1 / period_len) - 1)

    @staticmethod
    def calculate_sfr(fcr: np.ndarray, sgr: np.ndarray):
        return fcr * sgr
    
    def calculate_factors(self,
                          start_period_name: str,
                          start_df: pd.DataFrame,
                          end_period_name: str,
                          end_df: pd.DataFrame):        
        fish_count_end = end_df['close_count'].values
        fish_count_start = start_df['close_count'].values
        
        average_weight_end = end_df['close_weight_g'].values
        average_weight_start = start_df['close_weight_g'].values
        
        sum_feed_end = end_df[f'{start_period_name}-{end_period_name}-sum'].values
        sum_feed_start = start_df[f'{start_period_name}-{end_period_name}-sum'].values
        assert (sum_feed_end == sum_feed_start).all()
        sum_feed = sum_feed_end
        
        period_len_end = end_df[f'{start_period_name}-{end_period_name}-len'].values
        period_len_start = start_df[f'{start_period_name}-{end_period_name}-len'].values
        assert (period_len_end == period_len_start).all()
        period_len = period_len_end

        fcr = self.calculate_fcr(sum_feed, fish_count_end, average_weight_end, fish_count_start, average_weight_start)
        sgr = self.calculate_sgr(average_weight_end, average_weight_start, period_len)
        sfr = fcr * sgr

        return fcr, sgr, sfr

    def get_period_len(self,
                       df: pd.DataFrame,
                       period_end_col: str,
                       period_start_col: str):
        grouped = df.groupby('final_locus_population_id').mean(numeric_only=False)
        return (grouped[period_end_col] - grouped[period_start_col]).dt.days

    def get_period_sum(self,
                       df: pd.DataFrame,
                       period_end_col=None,
                       period_start_col=None,
                       key_col=''):
        
        period = df[df['event_date'] <= df[period_end_col]]
        if period_start_col is not None:
            period = period[df['event_date'] >= df[period_start_col]]
        return period.groupby('final_locus_population_id')['amount'].sum()

    def process(self, feed_df):

        agg_df = pd.DataFrame()
        agg_df['first_feeding-transfer-len'] = self.get_period_len(feed_df, 'shipout_date', 'first_feeding_date')
        agg_df['first_feeding-transfer-len'] = self.get_period_len(feed_df, 'shipout_date', 'first_feeding_date')
        agg_df['first_feeding-vaccination-len'] = self.get_period_len(feed_df,'VAC_EVENT_DATE', 'first_feeding_date')
        agg_df['vaccination-transfer-len'] = self.get_period_len(feed_df, 'shipout_date', 'VAC_EVENT_DATE')

        agg_df['first_feeding-transfer-sum'] = self.get_period_sum(feed_df,
                                                                    'shipout_date',
                                                                    'first_feeding_date',
                                                                    'amount')
        agg_df['first_feeding-vaccination-sum'] = self.get_period_sum(feed_df,
                                                                      'VAC_EVENT_DATE',
                                                                      'first_feeding_date',
                                                                      'amount')                                                            
        agg_df['vaccination-transfer-sum'] = self.get_period_sum(feed_df,
                                                                  'shipout_date',
                                                                  'VAC_EVENT_DATE',
                                                                  'amount')     
        agg_df.reset_index(inplace=True)

        first_feeding = feed_df[feed_df['event_date'] == feed_df['first_feeding_date']].reset_index(drop=True)
        vaccination = feed_df[feed_df['event_date'] == pd.to_datetime(feed_df['VAC_EVENT_DATE'].dt.date)]\
        .reset_index(drop=True)
        transfer = feed_df[feed_df['event_date'] == feed_df['shipout_date']].reset_index(drop=True)
        
        time_periods = {
            'start': [{'name': 'first_feeding', 'data': first_feeding.merge(agg_df)}, 
                      {'name': 'first_feeding', 'data': first_feeding.merge(agg_df)},
                      {'name': 'vaccination', 'data': vaccination.merge(agg_df)}],
            'end': [{'name': 'vaccination', 'data': vaccination.merge(agg_df)}, 
                    {'name': 'transfer', 'data': transfer.merge(agg_df)},
                    {'name': 'transfer', 'data': transfer.merge(agg_df)}]
        }
        
        for start, end in zip(time_periods['start'], time_periods['end']):
            fcr, sgr, sfr = self.calculate_factors(start['name'], start['data'], end['name'], end['data'])
            agg_df[f"{start['name']}-{end['name']}-fcr"] = fcr
            agg_df[f"{start['name']}-{end['name']}-sgr"] = sgr
            agg_df[f"{start['name']}-{end['name']}-sfr"] = sfr

        return agg_df
    
def create_factors_df(agg_ph_df, factors, key_columns, weight_column, weighted_func):
    """Creates factors df on key_columns level using weight_column to calculate weighted average"""
    factors_dfs = []
    print('Weighting factors')
    for factor in tqdm(factors):
        tmp = agg_ph_df.groupby(key_columns).apply(weighted_func, weight_column, factor).reset_index().rename(columns={0: factor})
        factors_dfs.append(tmp)

    factor_df = factors_dfs[0]
    for df in factors_dfs[1:]:
        factor_df = factor_df.merge(df, on=key_columns, how='inner')
    return factor_df
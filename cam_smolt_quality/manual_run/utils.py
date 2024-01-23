import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings

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


def read_file(filename, conn_params=None, buffer=False, save=True, encoding='utf-8'):
    print(f'Reading {filename}, from {"csv" if buffer else "sql"}')
    query_path = os.path.join(QUERIES_FILEPATH, filename+'.sql')
    csv_path = os.path.join(CSV_BUFFER_FILEPATH, filename+'.csv')

    if not buffer:
    
        with open(query_path, 'r', encoding=encoding) as file:
            query = file.read()
        connection_string = f"postgresql://{conn_params['user']}:{conn_params['password']}@{conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}"
        with create_engine(connection_string,  connect_args={'options': f'-c client_encoding={encoding}'}).connect() as engine:
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

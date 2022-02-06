import pandas as pd
import numpy as np
import glob
import os
import re
import datetime
import pickle

from dateutil.relativedelta import relativedelta
from collections import defaultdict

k_date_quarters = dict(
    一季 = 3,
    二季 = 6,
    三季 = 9,
    四季 = 12,
)

def translate_date(date: str):
    m = re.match(r'(\d{4})年(.*季)', date)
    if m:
        year, quarter = m.groups()
        month = k_date_quarters[quarter]
        return (datetime.datetime.strptime(f'{year}-{month}', '%Y-%m') +
                relativedelta(months = 1) + relativedelta(days = -1)).date()

    assert False, f'invalid date: {date}'

def eval_rank(rank):
    if isinstance(rank, str):
        a, b = rank.split('/')
        result = float(a) / float(b)
    elif isinstance(rank, datetime.datetime):
        a, b, c = rank.strftime('%m-%d-%Y').split('-')
        result = float(a) / float(b) / float(c)
    else:
        raise ValueError(f'invalid rank type: {rank}')
    assert result <= 1.0, f'invalid rank: {rank}'
    return result

def load_stocks_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = 1, **args)

    dfs = []
    for sheetname, data in df.items():
        assert all(data.columns[i] == f'Unnamed: {i}' for i in range(3)), f'invalid columns: {data.info()}'
        data = data.rename({
            'Unnamed: 0': 'stock_id',
            'Unnamed: 1': 'stock_name',
            'Unnamed: 2': 'indicator',
        }, axis=1).dropna(how='all')

        data = data[data['stock_id'].str.contains('数据来源：东方财富Choice数据') == False]
        data = data.melt(id_vars=data.columns.tolist()[:3]).dropna(how='any')
        data.columns = ['stock_id', 'stock_name', 'indicator', 'time', 'value']
        data.loc[:, ['time']] = data['time'].apply(lambda d : d.date())
        data = data.pivot(index = ['stock_id', 'stock_name', 'time'], columns = 'indicator', values = 'value')

        dfs.append(data)

        # print(f'  sheet: {sheetname}')
        # print(data.info())
        # print(data.head())
        # print(data.tail())

    data = pd.concat(dfs, axis = 0)
    assert data.index.duplicated().sum() == 0,  f'invalid data: {data.info()}'
    return data

def load_funds_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = 1, **args)

    dfs = []
    for sheetname, data in df.items():
        data = data.rename({
            'Unnamed: 0': 'fund_id',
            'Unnamed: 1': 'fund_name',
            'Unnamed: 2': 'indicator',
        }, axis=1).dropna(how='all')

        data = data[data['fund_id'].str.contains('数据来源：东方财富Choice数据') == False]
        data = data.melt(id_vars=data.columns.tolist()[:3]).dropna(how='any')
        data.columns = ['fund_id', 'fund_name', 'indicator', 'time', 'value']
        data.loc[:, ['time']] = data['time'].apply(lambda d : d.date())
        data = data.pivot(index = ['fund_id', 'fund_name', 'time'], columns = 'indicator', values = 'value')

        dfs.append(data)

        # print(f'  sheet: {sheetname}')
        # print(data.info())
        # print(data.head())
        # print(data.tail())

    data = pd.concat(dfs, axis = 0)
    assert data.index.duplicated().sum() == 0,  f'invalid data: {data.info()}'
    return data

def load_funds_perf_summary_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = 1, header = [0, 1], **args)

    dfs = []
    for sheetname, data in df.items():
        data = data[data.columns[:-3]].dropna(how='any')
        data = data.melt(id_vars=data.columns.tolist()[:2])
        data.columns = ['fund_id', 'fund_name', 'time', 'indicator', 'value']
        data['time'] = data['time'].apply(translate_date)

        pred = data['indicator'] == '同类排名'
        data.loc[pred, ['value']] = data[pred]['value'].apply(eval_rank)
        data = data.pivot(index = ['fund_id', 'fund_name', 'time'], columns = 'indicator', values = 'value')

        dfs.append(data)

    data = pd.concat(dfs, axis = 0).sort_index()
    assert data.index.duplicated().sum() == 0, f'invalid data: {data.info()}'
    return data

    # print('=== funds performance summary ===')
    # print(data.info())
    # print(data.head())
    # print(data.tail())

def load_funds_topn_stocks_detail_and_preprocess(file, args = {}):
    pass

def preprocess_wrapper(processor, *args):
    try:
        return processor(*args)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise e

def postprocess(dfs):
    grouped_by_index = defaultdict(list)
    for df in dfs:
        if df is None: continue
        index = tuple(df.index.names)
        grouped_by_index[index].append(df)

    result = {}
    for index, dfs in grouped_by_index.items():
        df = pd.concat([df.reset_index() for df in dfs],
                        axis = 0,
                        join = 'outer',
                        ignore_index = True).reset_index(drop=True)

        df = df.groupby(list(index)).first()
        df = df.dropna(axis = 1, how='all')[sorted(df.columns)]
        result[index] = df

        # ind = '  '
        # print(f'index: {index}')
        # print(f'columns: {len(df.columns)}')
        # print(ind + f'\n{ind}'.join(df.columns))

    return result

def load_data(args):
    all_data_args = []
    for file in glob.glob(os.path.join(args.path, '**/*.xlsx'), recursive=True):
        dirname  = os.path.dirname(file)
        basename = os.path.basename(file)

        m = re.match(f'(.*?).xlsx.*', basename)
        assert m, f'invalid file name: {file}'

        data_args = None
        name = m.group(1)
        if   name.startswith('stock'):
            data_args = [load_stocks_and_preprocess, file, {}]
        elif name.startswith('funds.perf_indicator') or \
             name.startswith('funds.value_indicator'):
            data_args = [load_funds_and_preprocess, file, {}]
        elif name.startswith('funds.perf_summary'):
            data_args = [load_funds_perf_summary_and_preprocess, file, {}]
        elif name.startswith('funds.topn_stocks_detail'):
            data_args = [load_funds_topn_stocks_detail_and_preprocess, file, {}]
        else:
            print(f'skip file: {file}')

        if data_args:
            all_data_args.append(data_args)

    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            result = pool.starmap(preprocess_wrapper, all_data_args)
    else:
        result = []
        for data_args in all_data_args:
            result.append(preprocess_wrapper(*data_args))

    result = postprocess(result)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'raw_data.pkl'), 'wb') as f:
        pickle.dump(result, f)

    return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',           default='raw_data')
    parser.add_argument('--output', '-o',   default=f'output_{datetime.datetime.now().isoformat(timespec="seconds")}')
    parser.add_argument('--jobs',   '-j',   default=1, type=int)

    args = parser.parse_args()

    load_data(args)

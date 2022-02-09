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
    中   = 6,
    年   = 12,
)

def parse_multidim_column(name: str):
    name = name.replace('_x000D_', ' ')

    result_name = []
    result_time = None
    for tok in name.split():
        m = re.match(r'\[(.+?)\](\S+)', tok)
        if m:
            key, value = m.groups()
            if re.match(r'(\d{4})-(\d{2}-(\d{2}))', value):
                result_time = datetime.datetime.strptime(value, '%Y-%m-%d').date()
            else:
                result_name.append(tok)
        else:
            result_name.append(tok)

    return ' '.join(result_name), result_time

def translate_date(date: str):
    m = re.match(r'(\d{4})年(.*季)', date)
    if m:
        year, quarter = m.groups()
        month = k_date_quarters[quarter]
        return (datetime.datetime.strptime(f'{year}-{month}', '%Y-%m') +
                relativedelta(months = 1) + relativedelta(days = -1)).date()

    m = re.match(r'(\d{4})年(.*)报', date)
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
    return [data]

def load_stocks_multidim_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = [0, 2], **args)

    dfs_timed   = []
    dfs_untimed = []
    for sheetname, data in df.items():
        assert all(data.columns[i] == f'Unnamed: {i}' for i in range(2)), f'invalid columns: {data.info()}'
        data = data.rename({
            'Unnamed: 0': 'stock_id',
            'Unnamed: 1': 'stock_name',
        }, axis=1).dropna(how='all')

        columns_timed       = []
        columns_timed_new   = []
        columns_untimed     = []
        columns_untimed_new = []
        for c in data.columns[2:]:
            name, time = parse_multidim_column(c)
            if time is not None:
                columns_timed.append(c)
                columns_timed_new.append((name, time))
            else:
                columns_untimed.append(c)
                columns_untimed_new.append(name)

        index = ['stock_id', 'stock_name']
        if columns_timed:
            df = data[index + columns_timed]
            df.columns = pd.MultiIndex.from_tuples([(c, '') for c in data.columns.tolist()[:2]] + columns_timed_new)

            df = df.melt(id_vars=df.columns.tolist()[:2]).dropna(how='any')
            df.columns = ['stock_id', 'stock_name', 'indicator', 'time', 'value']
            df = df.pivot(index = ['stock_id', 'stock_name', 'time'], columns = 'indicator', values = 'value')

            dfs_timed.append(df)

            # print(f'  sheet: {sheetname} timed:')
            # print(df.info())
            # print(df.head())
            # print(df.tail())

        if columns_untimed:
            df = data[index + columns_untimed]
            df.columns = index + columns_untimed_new

            df = df.melt(id_vars=df.columns.tolist()[:2]).dropna(how='any')
            df.columns = ['stock_id', 'stock_name', 'indicator', 'value']
            df = df.pivot(index = ['stock_id', 'stock_name'], columns = 'indicator', values = 'value')

            dfs_untimed.append(df)

            # print(f'  sheet: {sheetname} untimed:')
            # print(df.info())
            # print(df.head())
            # print(df.tail())

    result = []
    if dfs_timed:
        result.append(pd.concat(dfs_timed, axis = 0))
        assert result[-1].index.duplicated().sum() == 0, f'invalid data: {result[-1].info()}'

    if dfs_untimed:
        result.append(pd.concat(dfs_untimed, axis = 0))
        assert result[-1].index.duplicated().sum() == 0, f'invalid data: {result[-1].info()}'

    return result

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
    return [data]

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
    return [data]

    # print('=== funds performance summary ===')
    # print(data.info())
    # print(data.head())
    # print(data.tail())

def load_funds_topn_stocks_detail_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = 1, **args)

    dfs = []
    for sheetname, data in df.items():
        data = data[data.columns[:-2]].dropna(how='all')
        data = data[data['代码'].str.contains('数据来源：东方财富Choice数据') == False]

        index = ['fund_id', 'fund_name', 'time', 'stock_id', 'stock_name']

        data = data.melt(id_vars=data.columns.tolist()[:5]).dropna(how='any')
        data.columns = index + ['indicator', 'value']
        data.loc[:, ['time']] = data['time'].apply(translate_date)
        data = data.pivot(index = index, columns = 'indicator', values = 'value')

        dfs.append(data)

    data = pd.concat(dfs, axis = 0).sort_index()
    assert data.index.duplicated().sum() == 0, f'invalid data: {data.info()}'
    return [data]

    # print('=== funds topn stocks summary ===')
    # print(data.info())
    # print(data.head())
    # print(data.tail())

def load_funds_multidim_and_preprocess(file, args = {}):
    print(f'loading {file} with {args}')
    df = pd.read_excel(file, sheet_name = None, skiprows = [0, 2], **args)

    dfs_timed   = []
    dfs_untimed = []
    for sheetname, data in df.items():
        assert all(data.columns[i] == f'Unnamed: {i}' for i in range(2)), f'invalid columns: {data.info()}'
        data = data.rename({
            'Unnamed: 0': 'fund_id',
            'Unnamed: 1': 'fund_name',
        }, axis=1).dropna(how='all')

        columns_timed       = []
        columns_timed_new   = []
        columns_untimed     = []
        columns_untimed_new = []
        for c in data.columns[2:]:
            name, time = parse_multidim_column(c)
            if time is not None:
                columns_timed.append(c)
                columns_timed_new.append((name, time))
            else:
                columns_untimed.append(c)
                columns_untimed_new.append(name)

        index = ['fund_id', 'fund_name']
        if columns_timed:
            df = data[index + columns_timed]
            df.columns = pd.MultiIndex.from_tuples([(c, '') for c in data.columns.tolist()[:2]] + columns_timed_new)

            df = df.melt(id_vars=df.columns.tolist()[:2]).dropna(how='any')
            df.columns = ['fund_id', 'fund_name', 'indicator', 'time', 'value']
            df = df.pivot(index = ['fund_id', 'fund_name', 'time'], columns = 'indicator', values = 'value')

            dfs_timed.append(df)

            # print(f'  sheet: {sheetname} timed:')
            # print(df.info())
            # print(df.head())
            # print(df.tail())

        if columns_untimed:
            df = data[index + columns_untimed]
            df.columns = index + columns_untimed_new

            df = df.melt(id_vars=df.columns.tolist()[:2]).dropna(how='any')
            df.columns = ['fund_id', 'fund_name', 'indicator', 'value']
            df = df.pivot(index = ['fund_id', 'fund_name'], columns = 'indicator', values = 'value')

            dfs_untimed.append(df)

            # print(f'  sheet: {sheetname} untimed:')
            # print(df.info())
            # print(df.head())
            # print(df.tail())

    result = []
    if dfs_timed:
        result.append(pd.concat(dfs_timed, axis = 0))
        assert result[-1].index.duplicated().sum() == 0, f'invalid data: {result[-1].info()}'

    if dfs_untimed:
        result.append(pd.concat(dfs_untimed, axis = 0))
        assert result[-1].index.duplicated().sum() == 0, f'invalid data: {result[-1].info()}'

    return result



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
        if   name.startswith('stock.shareholder_ratio') or \
             name.startswith('stock.avg_value') or \
             name.startswith('stock.static_info') or \
             name.startswith('stock.dynamic_indicators') or \
             name.startswith('stock.static_indicators'):
            data_args = [load_stocks_multidim_and_preprocess, file]
        elif name.startswith('stock'):
            data_args = [load_stocks_and_preprocess, file]
        elif name.startswith('funds.static_info') or \
             name.startswith('funds.dynamic_indicators'):
            data_args = [load_funds_multidim_and_preprocess, file]
        elif name.startswith('funds.perf_indicator') or \
             name.startswith('funds.value_indicator'):
            data_args = [load_funds_and_preprocess, file]
        elif name.startswith('funds.perf_summary'):
            data_args = [load_funds_perf_summary_and_preprocess, file]
        elif name.startswith('funds.topn_stocks_detail'):
            data_args = [load_funds_topn_stocks_detail_and_preprocess, file]
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

    result = postprocess([r for res in result for r in res])

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

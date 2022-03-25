import json
import pandas as pd


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def investing_yield_data(path):
    yield_data = read_csv(path)
    yield_data = yield_data[yield_data.columns[0:-1]]
    yield_data.columns = ['Date', 'Close', 'Open', 'High', 'Low']
    yield_data['Date'] = pd.to_datetime(yield_data['Date'], format='%d.%m.%Y')
    for col in yield_data.columns[1:]:
        yield_data[col] = yield_data[col].str.replace(',', '.').str.replace('%', '')
        yield_data[col] = yield_data[col].astype(float) / 100.0
    return yield_data
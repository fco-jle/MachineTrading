import pandas as pd
import datetime as dt
import requests
from bs4 import BeautifulSoup
import json
import os

filepath = os.path.join(os.path.dirname(__file__), "Eurex_Futures_URLS.json")
with open(filepath, 'r') as f:
    EUREX_URL = json.load(f)

def settlement_price(instrument:str, delivery_date:dt.date):
    assert instrument in EUREX_URL.keys()
    dfs = pd.read_html(EUREX_URL[instrument]['URL'])
    price_df = dfs[0].copy()
    formatted_date = delivery_date.strftime("%b %y")
    settle_price = price_df[price_df['Delivery month']==formatted_date].iloc[0]['Settlem. price']
    return settle_price

if __name__ == '__main__':
    import prettytable as pt
    prices = []
    next_delivery = dt.date(2022,6,10)

    for contract in EUREX_URL.keys():
        price = settlement_price(instrument=contract, delivery_date=next_delivery)
        name = EUREX_URL[contract]['Name']
        prices.append((name, contract, price))

    prices = sorted(prices, key=lambda x: x[1])

    tab = pt.PrettyTable(['Ticker', 'Contract', 'Settle Price'])
    for x in prices:
        tab.add_row([x[1], x[0], x[2]])
    tab.float_format = '.4'

    display(tab)
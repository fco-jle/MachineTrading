import pandas as pd
import datetime as dt
import requests
from bs4 import BeautifulSoup
import json
import os

def update_eurex_urls():
    eurex_main_url = r"https://www.eurex.com/ex-en/markets/int/fix/government-bonds"
    page = requests.get(eurex_main_url.strip())
    soup = BeautifulSoup(page.content, "html.parser")
    urls_dict = {}
    boxes = soup.find_all("div", {"class": "dbx-search-result"})
    for b in boxes:
        links = b.find_all('a', {"class": "dbx-search-result__title-url"})
        for a in links:
            name = a.text.strip()
            partial_url = a['href']
            ticker = name.split()[-1][1:-1]
            urls_dict[ticker] = {"URL":eurex_main_url + '/' + partial_url.split('/')[-1],
                                 "Name":name}
            print("Found URL:", ticker, partial_url)

    filepath = os.path.join(os.path.dirname(__file__), "Eurex_Futures_URLS.json")
    with open(filepath, 'w') as fp:
        json.dump(urls_dict, fp, indent=4)

if __name__ == '__main__':
    update_eurex_urls()

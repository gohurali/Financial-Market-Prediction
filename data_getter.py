import requests
import yaml
import json
import numpy as np
import pandas as pd
import torch

def fetch_btc_prices():
    with open("BTC_data.json") as f:
        return json.load(f)

def parse_alphaV_JSON(raw_data):
    raw_data.pop('Meta Data',None)

    df = pd.DataFrame.from_dict(raw_data['Time Series (Digital Currency Daily)'])
    df = df.transpose()
    print(df)

    pass

def main():
    raw_price_data = fetch_btc_prices()
    parse_alphaV_JSON(raw_data=raw_price_data)
    

    pass

if __name__ == '__main__':
    main()
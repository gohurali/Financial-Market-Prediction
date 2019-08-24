from flask import Flask
from flask import jsonify
from flask import request
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.architectures import TimeRNN
from inference import Inferencer
import time
import sys
import os
import numpy as np
import requests
import pandas as pd
from utils.preprocessing import MinMaxScaler
"""
Flask JSON Endpoint Webservice

Takes input from post request and preprocesses
the data by post processing it
"""

app = Flask(__name__)

def get_config(file_loc='config.yaml'):
    return yaml.safe_load(open(file_loc))

config = get_config()

def load_model():
    model_path = config['model_save_loc']
    model = TimeRNN(bat_size=1,in_features=3,h_size=1,layer_amnt=1)
    model.load_state_dict(torch.load(config['model_save_loc']))
    model.eval()
    return model

def preprocess(input_data):
    # -- Normalize --
    inf = Inferencer()
    raw_data = inf.fetch_latest_BTC_JSON()
    df = inf.parse_alphaV_JSON(raw_data=raw_data)
    prices = np.array(df['4a. close (USD)'].tolist())
    data_df_temp = df.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1)
    minmax_2 = MinMaxScaler(data_df_temp.values)
    data_df_temp = pd.DataFrame(minmax_2.fit_transform(), columns=data_df_temp.columns)

    minimum_price = np.min(prices)
    maximum_price = np.max(prices)
    return input_data,minmax_2,minimum_price,maximum_price

def postprocess(result):
    
    result = np.array(result).item()
    
    inf = Inferencer()
    
    raw_data = inf.fetch_latest_BTC_JSON()
    df = inf.parse_alphaV_JSON(raw_data=raw_data)
    prices = np.array(df['4a. close (USD)'].tolist())

    minimum_price = np.min(prices)
    maximum_price = np.max(prices)

    res = inf.un_normalize(norm_val=result,min_val=minimum_price,max_val=maximum_price)
    return res

@app.route("/",methods=['POST'])
def inference():
    input_data = request.get_json(force=True)
    model = load_model()
    input_data = input_data['data']
    

    start = time.time()   # start timer
    input_data,normalizer,min_price,max_price = preprocess(input_data) 
    inf = Inferencer()
    res = inf.inference(value=input_data,
                        normalize_method=normalizer,
                        model=model,
                        minimum_price=min_price,
                        maximum_price=max_price
                        )
    for idx,arr in enumerate(res):
        res[idx] = arr.numpy().item()
    end = time.time()     # stop timer
    return jsonify({"result": res,"time": end - start})

if __name__ == '__main__':
    app.run(debug=True)
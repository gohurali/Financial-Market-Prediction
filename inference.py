import requests
import os
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import urllib3
import cryptocompare
from datetime import datetime
from models.architectures import TimeRNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config(file_loc='config.yaml'):
    return yaml.safe_load(open(file_loc))

config = get_config()

class Inferencer(object):
    def __init__(self):
        self.model = self.open_model()
    
    def open_model(self):
        model = TimeRNN(bat_size=1,in_features=3,h_size=1,layer_amnt=1)
        model.load_state_dict(torch.load(config['model_save_loc']))
        model.eval()
        return model

    def un_normalize(self,norm_val,min_val,max_val,typelist=None):
        if(typelist):
            for idx,item in enumerate(norm_val):
                new_val = item * (max_val - min_val) + min_val
                norm_val[idx] = new_val
            return norm_val
        else:
            return norm_val * (max_val - min_val) + min_val 

    def inference(self,value, normalize_method, model,minimum_price,maximum_price):
        value = np.array(value)
        predictions = []
        for sample in value:
            sample = np.array(sample).reshape(1,-1)
            example = torch.tensor(normalize_method.transform(sample)).float()
            
            if(str(device) == 'cuda'):
                example = example.to(device)

            output = model(example)
            output_unnorm = self.un_normalize(norm_val=output.detach(),min_val=minimum_price,max_val=maximum_price)
            predictions.append(output_unnorm)
        return predictions

    def fetch_latest_BTC_JSON(self):
        """Fetch the latest JSON data"""
        API_LINK = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=SAITMI5ZUMGEKGKY'
        page = requests.get(API_LINK).json()
        return page
    def parse_alphaV_JSON(self,raw_data):
        # Remove meta data for now
        raw_data.pop('Meta Data',None)
        # Remove key name
        df = pd.DataFrame.from_dict(raw_data['Time Series (Digital Currency Daily)'],dtype=float)
        # Flip dates as columns into rows
        df = df.transpose()
        return df



# class TimeRNN(nn.Module):
#     def __init__(self,bat_size,in_features,h_size,layer_amnt):
#         super(TimeRNN,self).__init__()
        
#         self.batch_sz = bat_size
#         self.in_features = in_features
#         self.h_size = h_size
#         self.layer_amnt = layer_amnt
        
#         self.lstm1 = nn.LSTM(input_size=self.in_features,
#                              hidden_size=self.h_size,
#                              num_layers=self.layer_amnt,
#                              bias=True,
#                              batch_first=True,
#                              dropout=0,
#                              bidirectional=False)
#         self.fc1 = nn.Linear(in_features=1,out_features=1)
#     def init_hidden(self):
#         """Intialize/re-init the hidden and cell states. 
#         The hidden state acts as the memory of the RNN 
#         which gets passed from one unit to another. 
#         h_i = f(h_i + in)

#         Intializing with 0s
#         """
#         #print('layer size =\t', self.layer_amnt)
#         #print('bat_size =\t', self.batch_sz)
#         #print('hidden size =\t',self.h_size)
#         return (torch.zeros(self.layer_amnt,self.batch_sz,self.h_size),
#                 torch.zeros(self.layer_amnt,self.batch_sz,self.h_size))
#     def forward(self,x):
#         x = x.unsqueeze(0)
#         hidden_init = self.init_hidden()
#         h0 = hidden_init[0].to(device)
#         c0 = hidden_init[1].to(device)
#         x,hidden = self.lstm1( x,(h0,c0))
#         x = F.leaky_relu(self.fc1(x[-1].view(self.batch_sz,-1)))
#         return x


def main():
    inf = Inferencer()

    histPriceDay = cryptocompare.get_historical_price_day('BTC', curr='USD')

    # Getting CryptoCompare BTC volume data -- 2000 API calls back 
    vol = []
    for idx, item in enumerate(histPriceDay['Data']):
        vol.append(item['volumefrom']) 

    raw_data = inf.fetch_latest_BTC_JSON()
    df = inf.parse_alphaV_JSON(raw_data=raw_data)
    prices = np.array(df['4a. close (USD)'].tolist())
    data_df_temp = df.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1) # ,'2a. high (USD)','3a. low (USD)'
    minmax_2 = preprocessing.MinMaxScaler()
    data_df_temp = pd.DataFrame(minmax_2.fit_transform(data_df_temp), columns=data_df_temp.columns)

    minimum_price = np.min(prices)
    maximum_price = np.max(prices)

    output = inf.inference(value=[ [9400,9800,vol[-1]],
                                   [9400,10000,35000]],
                       normalize_method=minmax_2,
                       model=inf.model,
                       minimum_price=minimum_price,
                       maximum_price=maximum_price
                      )
    print('BTC prediction: ', output)
    
if __name__ == '__main__':
    main()
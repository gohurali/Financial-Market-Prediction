import numpy as np
import pandas as pd
import requests
import yaml
import json
from sklearn import preprocessing

class DataPrepper():
    def __init__(self):
        self.raw_data = self.fetch_latest_BTC_JSON()
        self.dataframe = self.parse_alphaV_JSON(raw_data=self.raw_data)
        self.prices = np.array(self.dataframe['4a. close (USD)'].tolist())

        # Temporary dataframe for creating an extra normalizer for re-scaling inference values later
        data_df_temp = self.dataframe.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1) # ,'2a. high (USD)','3a. low (USD)'
        self.minmax_2 = preprocessing.MinMaxScaler()
        data_df_temp = pd.DataFrame(self.minmax_2.fit_transform(data_df_temp), columns=data_df_temp.columns)
        
        # -- Normalize the Data --
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.dataframe = pd.DataFrame(self.min_max_scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
        self.dataframe = self.table_edit(self.dataframe)
        self.targets = np.array(self.dataframe['4a. close (USD)'].tolist())
        self.dataframe = self.dataframe.drop(labels=['4a. close (USD)'],axis=1)
        
        self.dataframe = self.dataframe.drop(labels=['1a. open (USD)','6. market cap (USD)'],axis=1)

    
    def get_data(self):
        """Returns x-data and y-data"""
        return self.dataframe.values,self.targets
    
    def fetch_latest_BTC_JSON(self):
        """Fetch the latest JSON data"""
        API_LINK = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey=SAITMI5ZUMGEKGKY'
        page = requests.get(API_LINK).json()
        return page
    def fetch_btc_prices(self):
        with open("/content/gdrive/My Drive/College/Undergraduate Research/StockData/BTC_data.json") as f:
            return json.load(f)  

    def parse_alphaV_JSON(self,raw_data):
        # Remove meta data for now
        raw_data.pop('Meta Data',None)
        # Remove key name
        df = pd.DataFrame.from_dict(raw_data['Time Series (Digital Currency Daily)'],dtype=float)
        # Flip dates as columns into rows
        df = df.transpose()
        return df
    
    def table_edit(self,dataframe):
        dataframe = dataframe.drop(labels=['1b. open (USD)','2b. high (USD)','3b. low (USD)','4b. close (USD)'],axis=1)
        table_col_order = ['1a. open (USD)','2a. high (USD)','3a. low (USD)','5. volume','6. market cap (USD)','4a. close (USD)']
        dataframe = dataframe[table_col_order]
        return dataframe


    
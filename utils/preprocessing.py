import numpy as np
import pandas as pd
import requests
import yaml
import json

class Feature:
    def __init__(self,data):
        self.data = data
        self.feat_max = self.calc_max()
        self.feat_min = self.calc_min()
    def calc_max(self):
        return np.max(self.data)
    def calc_min(self):
        return np.min(self.data)

class MinMaxScaler:
    def __init__(self,data):
        self.data = data
        self.original_data = data
        self.feature_col = []
    def normalize(self,val,min_val,max_val):
        return (val-min_val)/(max_val-min_val)
    def inverse_normalize(self,val,min_val,max_val):
        return val * (max_val - min_val) + min_val
    def fit_transform(self):
        for column in self.data.T:
            feat = Feature(column)
            self.feature_col.append(feat)
            for idx,value in enumerate(column):
                #print(value)
                column[idx] = self.normalize(val=value,min_val=feat.feat_min,max_val=feat.feat_max)
            #print()
        return self.data
    def inverse_transform(self,val):
        typelist=True
        val = np.array(val)
        if(typelist):
            for idx,(value,feature_col) in enumerate(zip(val.T,self.feature_col)):
                #print(value)
                #print(feature_col.feat_max)
                transform_val = self.inverse_normalize(val=value,min_val=feature_col.feat_min,max_val=feature_col.feat_max)
                val.T[idx] = transform_val
                #val[idx] = new_val
            return val
        else:
            return val * (max_val - min_val) + min_val  

class DataPrepper():
    def __init__(self):
        self.raw_data = self.fetch_latest_BTC_JSON()
        self.dataframe = self.parse_alphaV_JSON(raw_data=self.raw_data)
        self.prices = np.array(self.dataframe['4a. close (USD)'].tolist())

        # Temporary dataframe for creating an extra normalizer for re-scaling inference values later
        data_df_temp = self.dataframe.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1) # ,'2a. high (USD)','3a. low (USD)'
        #self.minmax_2 = preprocessing.MinMaxScaler()
        #data_df_temp = pd.DataFrame(self.minmax_2.fit_transform(data_df_temp), columns=data_df_temp.columns)
        self.minmax_2 = MinMaxScaler(data=data_df_temp.values)
        data_df_temp = pd.DataFrame(self.minmax_2.fit_transform(), columns=data_df_temp.columns)

        # -- Normalize the Data --
        #self.min_max_scaler = preprocessing.MinMaxScaler()
        self.min_max_scaler = MinMaxScaler(data=self.dataframe.values)    

        #self.dataframe = pd.DataFrame(self.min_max_scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
        self.dataframe = pd.DataFrame(self.min_max_scaler.fit_transform(), columns=self.dataframe.columns)

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

def train_test_split(x_data,y_data,test_size):
    if(len(x_data) == len(y_data)):
        train_size = 1.0 - test_size
        train_idx = int(len(x_data)*train_size)
        x_train = x_data[:train_idx]
        x_test = x_data[train_idx:]
        y_train = y_data[:train_idx]
        y_test = y_data[train_idx:]
        return x_train,x_test,y_train,y_test
    else:
        raise Exception("x-data and y-data are of different sizes!")


# def main():
#     dp = DataPrepper()
#     #print(dp.dataframe)

#     pass

# if __name__ == '__main__':
#     main()
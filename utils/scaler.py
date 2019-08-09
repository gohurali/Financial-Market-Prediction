import numpy as np
from preprocessing import DataPrepper
from sklearn import preprocessing
import pandas as pd

class Feature:
    def __init__(self,data):
        self.data = data
        self.feat_max = self.calc_max()
        self.feat_min = self.calc_min()

    def calc_max(self):
        return np.max(self.data)
    def calc_min(self):
        return np.min(self.data)

class Scaler:


    def __init__(self,data):
        self.data = data
        self.original_data = data
        self.feature_col = []

    def normalize(self,val,min_val,max_val):
        return (val-min_val)/(max_val-min_val)
    def inverse_normalize(self,val,min_val,max_val):
        return val * (max_val - min_val) + min_val
    def fit(self):
        for column in self.data.T:
            feat = Feature(column)
            self.feature_col.append(feat)
            for idx,value in enumerate(column):
                #print(value)
                column[idx] = self.normalize(val=value,min_val=feat.feat_min,max_val=feat.feat_max)
            #print()
    
    def transform(self):
        pass

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


def main():
    dp = DataPrepper()
    raw_data = dp.fetch_latest_BTC_JSON()
    data = dp.parse_alphaV_JSON(raw_data=raw_data)
    prices = np.array(data['4a. close (USD)'].tolist())
    data = data.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1)
    print(data.values)
    print(data.values.T)
    #print(data.values[:,0])
    #print(data.values[:,1])

    sc = Scaler(data.values)
    sc.fit()
    print(sc.data)

    test_val = [[0.61461477, 0.59540211, 0.11104393],[0.0143552,0.01384498,0.10051906]]

    new_val = sc.inverse_transform(test_val)
    print(new_val)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # dataframe = pd.DataFrame(min_max_scaler.fit_transform(data), columns=data.columns)
    # print(dataframe.values)

if __name__ == '__main__':
    main()
import requests
import yaml
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
"""
Basic multilayer perceptron subclass 
of three layers in PyTorch
"""
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features=5,out_features=32)
        self.fc2 = nn.Linear(in_features=32,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=1)
    
    def forward(self, x):
        """"Forward pass definition"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

def fetch_btc_prices():
    with open("BTC_data.json") as f:
        return json.load(f)

def parse_alphaV_JSON(raw_data):
    # Remove meta data for now
    raw_data.pop('Meta Data',None)
    # Remove key name
    df = pd.DataFrame.from_dict(raw_data['Time Series (Digital Currency Daily)'],dtype=float)
    # Flip dates as columns into rows
    df = df.transpose()
    return df

def normalize():
    pass


def data_split(x_train,y_train):
        x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=100)
        return x_train, x_test, y_train, y_test

def train(model, x_train,y_train):
    batch_size = 8

    x_train, x_test, y_train, y_test = data_split(x_train,y_train)

    # Prep input data --> torch.float64
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    print(x_train.dtype)
    print(y_train.dtype)
    print(x_test.shape)
    print(x_test.shape)

    print('-- Model Architecture --')
    print(model)

    # -- Since we are predicting prices --> mean squared error is our loss function
    loss_func = torch.nn.MSELoss()

    # -- Optimizer --> Adam generally works best
    # TODO: choose a better learning rate later
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):

        y_predictions = model(x_train.float())
        loss = loss_func(y_predictions.float(),y_train.float())

        # zero out the gradients
        optimizer.zero_grad()
        # back-prop to update the weights
        loss.backward()
        # optimizer steps based on lr
        optimizer.step()

        if epoch % 100 == 0:
            print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, 100, loss.data[0].item()))


def table_edit(dataframe):
    dataframe = dataframe.drop(labels=['1b. open (USD)','2b. high (USD)','3b. low (USD)','4b. close (USD)'],axis=1)
    table_col_order = ['1a. open (USD)','2a. high (USD)','3a. low (USD)','5. volume','6. market cap (USD)','4a. close (USD)']
    dataframe = dataframe[table_col_order]
    return dataframe

def main():
    raw_price_data = fetch_btc_prices()
    data_df = parse_alphaV_JSON(raw_data=raw_price_data)

    data_df = table_edit(data_df)
    y_train = np.array(data_df['4a. close (USD)'].tolist())
    data_df = data_df.drop(labels=['4a. close (USD)'],axis=1)
    print(data_df)

    model = MLP()

    train(model, data_df.values,y_train)

    pass

if __name__ == '__main__':
    main()
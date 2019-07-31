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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("Device state:\t", device)
#print("Device index:\t",torch.cuda.current_device())
#print("Current device:\t", torch.cuda.get_device_name(device))
"""
Basic multilayer perceptron subclass 
of three layers in PyTorch
"""
class MLP(nn.Module):
    def __init__(self,num_features):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features=num_features,out_features=32)
        self.fc2 = nn.Linear(in_features=32,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=1)
    
    def forward(self, x):
        """"Forward pass definition"""
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class NumberRegression_MLP(nn.Module):
    def __init__(self):
        super(NumberRegression_MLP,self).__init__()
        self.fc1 = nn.Linear(in_features=1,out_features=32)
        self.fc2 = nn.Linear(in_features=32,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=1)
    def forward(self, x):
        """"Forward pass definition"""
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
"""
Recurrent Neural Networks (RNN)
RNNs are an excellent method to model sequential data
and time series data. This is perfect for stocks and
crytocurrency price prediction.

https://stackoverflow.com/questions/45022734/understanding-a-simple-lstm-pytorch
https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
"""
class TestRNN(nn.Module):
    def __init__(self,bat_size,in_features,h_size,layer_amnt):
        super(TestRNN,self).__init__()
        
        self.batch_sz = bat_size
        self.in_features = in_features
        self.h_size = h_size
        self.layer_amnt = layer_amnt
        
        self.lstm1 = nn.LSTM(input_size=self.in_features,
                             hidden_size=self.h_size,
                             num_layers=self.layer_amnt,
                             bias=True,
                             batch_first=True,
                             dropout=0,
                             bidirectional=False
                            )
    def init_hidden(self):
        """Intialize/re-init the hidden and cell states. 
        The hidden state acts as the memory of the RNN 
        which gets passed from one unit to another. 
        h_i = f(h_i + in)

        Intializing with 0s
        """
        #print('layer size =\t', self.layer_amnt)
        #print('bat_size =\t', self.batch_sz)
        #print('hidden size =\t',self.h_size)
        return (torch.zeros(self.layer_amnt,self.batch_sz,self.h_size),
                torch.zeros(self.layer_amnt,self.batch_sz,self.h_size))
    def forward(self,x):
        x = x.unsqueeze(0)
        hidden_init = self.init_hidden()
        h0 = hidden_init[0].to(device)
        c0 = hidden_init[1].to(device)
        x,hidden = self.lstm1( x,(h0,c0))
        return x
      
"""
Temporal Convolutional Neural Network (CNN)
Time series based convolutional neural network
"""
class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=256,kernel_size=2,stride=1,padding=0)
        self.fc1 = nn.Linear(in_features=512,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=1)
    def forward(self,x):
        x = x.unsqueeze(0)
        x = F.leaky_relu(self.conv1(x))
        x = x.reshape(-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

"""
https://arxiv.org/pdf/1506.00019.pdf
https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell
"""
class TimeRNN(nn.Module):
    def __init__(self,bat_size,in_features,h_size,layer_amnt):
        super(TimeRNN,self).__init__()
        
        self.batch_sz = bat_size
        self.in_features = in_features
        self.h_size = h_size
        self.layer_amnt = layer_amnt
        
        self.lstm1 = nn.LSTM(input_size=self.in_features,
                             hidden_size=self.h_size,
                             num_layers=self.layer_amnt,
                             bias=True,
                             batch_first=True,
                             dropout=0,
                             bidirectional=False)
        self.fc1 = nn.Linear(in_features=1,out_features=1)
    def init_hidden(self):
        """Intialize/re-init the hidden and cell states. 
        The hidden state acts as the memory of the RNN 
        which gets passed from one unit to another. 
        h_i = f(h_i + in)

        Intializing with 0s
        """
        #print('layer size =\t', self.layer_amnt)
        #print('bat_size =\t', self.batch_sz)
        #print('hidden size =\t',self.h_size)
        return (torch.zeros(self.layer_amnt,self.batch_sz,self.h_size),
                torch.zeros(self.layer_amnt,self.batch_sz,self.h_size))
    def forward(self,x):
        x = x.unsqueeze(0)
        hidden_init = self.init_hidden()
        h0 = hidden_init[0].to(device)
        c0 = hidden_init[1].to(device)
        x,hidden = self.lstm1( x,(h0,c0))
        x = F.leaky_relu(self.fc1(x[-1].view(self.batch_sz,-1)))
        return x

class TickerData(torch.utils.data.Dataset):
    def __init__(self, table):
        self.dataset = table
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        """idx --> data pointer"""
        return self.dataset[idx]

def get_config(file_loc='config.yaml'):
    return yaml.safe_load(open(file_loc))

config = get_config()

class Trainer(object):
    def __init__(self):
        pass
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
    
    def data_split(self,x_train,y_train):
        x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=100,shuffle=False)
        return x_train, x_test, y_train, y_test

    def train(self,model, x_data,y_data, original_prices,epochs):
        """Price prediction model training loop function. This method
        is generalized for the purposes of allowing any model to be used.

        Arguments:
            * model - torch subclassed model
            * x_data - input tensor dataset
            * y_data - targer tensor dataset
            * original_prices - original target dataset for re-scaling
            * epochs - number of epochs for training
        Returns:
            * losses
            * test_data_loader
            * loss_func
            * model 
            * min_price
            * max_price
            * y_test
        """
        prices = torch.tensor(original_prices)
        max_price = torch.max(prices)
        min_price = torch.min(prices)

        print('----Dataset Prep----')
        x_train, x_test, y_train, y_test = self.data_split(x_data,y_data)
        train_tensorDataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train))
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_tensorDataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        test_tensorDataset = torch.utils.data.TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_tensorDataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        print('-- Model Architecture --')
        print(model)

        # -- Model to CUDA GPU --
        if( str(device) == 'cuda'):
            print('Sending model to',torch.cuda.get_device_name(device),' GPU')
            #model = model.cuda()
            model.to(device)

        # -- Since we are predicting prices --> mean squared error is our loss function
        loss_func = torch.nn.MSELoss()

        # -- Optimizer --> Adam generally works best
        # TODO: choose a better learning rate later
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        total_loss = 0
        losses = []
        for epoch in (range(epochs)):
            for i, (examples,labels) in tqdm(enumerate(train_data_loader)):

                if( str(device) == 'cuda'):
                    examples = examples.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                y_predictions = model(examples.float())
                loss = loss_func(y_predictions.float(),labels.view(1,1).float())

                total_loss += loss.data

                # back-prop to update the weights
                loss.backward()
                # optimizer steps based on lr
                optimizer.step()

                y_preds = y_predictions.cpu().detach().numpy()
                y_preds = torch.tensor(y_preds)
                #print(y_preds.shape)
                test = self.un_normalize(y_preds, min_price, max_price)
                #print("---> ", test)
                #break

            print ('Epoch [{}/{}], Loss: {}'.format(epoch+1, config['epochs'], loss.data))
            #print(list(model.parameters()))
            print("-----------------------------------------------------------------------------")
            losses.append(loss.data)
        return losses, test_data_loader, loss_func, model, min_price, max_price, y_test


    def table_edit(self,dataframe):
        dataframe = dataframe.drop(labels=['1b. open (USD)','2b. high (USD)','3b. low (USD)','4b. close (USD)'],axis=1)
        table_col_order = ['1a. open (USD)','2a. high (USD)','3a. low (USD)','5. volume','6. market cap (USD)','4a. close (USD)']
        dataframe = dataframe[table_col_order]
        return dataframe

    def loss_visualize(self,loss_tensor):
        losses = np.array(loss_tensor)
        print(losses)

        plt.plot(losses)

    def validation_test(self,test_dataloader, criterion, model, norm_min, norm_max):
        test_loss = 0
        accuracy = 0
        all_predictions = []
        for (examples, labels) in test_dataloader:

            if(str(device) == 'cuda'):
                examples = examples.to(device)
                labels = labels.to(device)

            output = model.forward(examples)

            un_normed_outputs = self.un_normalize(output, norm_min,norm_max)
            all_predictions.append(un_normed_outputs.detach())
            #print("output --> ", un_normed_outputs)
            loss = criterion(output, labels.view(1,1)).item()
            test_loss += loss

            #ps = torch.exp(output)
            #equality = (labels.data == ps.max(dim=1)[1])
            #accuracy += equality.type(torch.FloatTensor).mean()

            #print('output --> ', un_normed_outputs, ' loss --> ', loss)

        return test_loss, accuracy, all_predictions

    def un_normalize(self,norm_val,min_val,max_val,typelist=None):
        if(typelist):
            for idx,item in enumerate(norm_val):
                new_val = item * (max_val - min_val) + min_val
                norm_val[idx] = new_val
            return norm_val
        else:
            return norm_val * (max_val - min_val) + min_val  

    def prediction_visualization(self,minimum_price,maximum_price,close_prices,model_predictions):
        plt.close()

        test_values = np.array(self.un_normalize(norm_val=close_prices,min_val=minimum_price,max_val=maximum_price,typelist=True))
        print(test_values)
        #print(np.array(model_predictions.grad))
        #model_predictions[0].requires_grad = False
        print(model_predictions) ###
        print(len(model_predictions))
        for idx,item in enumerate(model_predictions):
            print('the item is =\t',item)
            print('the size of the item is =\t', item.shape)
            model_predictions[idx] = np.asscalar(item.cpu().numpy())
        predicted_values = model_predictions

        print(len(test_values))
        print(len(predicted_values))

        plt.plot(np.array(test_values),color='#FFA500')
        plt.plot(np.array(predicted_values),color='g')
        plt.show()
        pass

    def volume_visualization(self,volume):
        vol = np.array(volume)
        plt.plot(vol)
        pass

    def VMA_calculation(self,prices=[]):
        prices = np.array(prices)

        n_prices = []
        for idx, p in enumerate(prices):
            if(idx + 1 == len(prices)):
                break
            p1 = prices[idx]
            p2 = prices[idx+1]
            arr = np.array([p1,p2])#,p3,p4,p5,p6,p7,p8,p9,p10,p11])
            n_prices.append(arr)

        vmas = []
        for arr in n_prices:
            sum_vol = arr.sum()
            vmas.append(sum_vol/2)

        first_ten = vmas[:10]
        average_num = np.average(first_ten)
        vmas = np.insert(vmas,0,average_num)

        #plt.figure(figsize=(20,10))
        #plt.plot(np.array(prices))
        #plt.plot(np.array(vmas))
        return vmas


    def vol_dataset_prep(self,vma,volume,original_volume):
        """VMA - x_train -- volume -- y_train"""
        print(len(vma))
        print(len(volume))
        print(len(original_volume))

    #     vma = torch.tensor(vma)
    #     volume = torch.tensor(volume)


        #########################
        x_train,x_test,y_train,y_test = data_split(volume,volume)

        #######################

        #x_train,x_test,y_train,y_test = data_split(vma,volume)

        train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train))
        test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test),torch.from_numpy(y_test))
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        return x_train,y_train,x_test,y_test,train_dataloader,test_dataloader

    def vol_train(self,model, train_dataloader):
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

        total_loss = 0
        losses = []
        for epoch in (range(10)):
            for i, (examples,labels) in tqdm(enumerate(train_dataloader)):

                optimizer.zero_grad()

                y_predictions = model(examples.float())
                loss = loss_func(y_predictions.float(),labels.float())

                total_loss += loss.data

                # back-prop to update the weights
                loss.backward()
                # optimizer steps based on lr
                optimizer.step()

                y_preds = y_predictions.cpu().detach().numpy()
                y_preds = torch.tensor(y_preds)
                #print(y_preds.shape)
                #print("---> ", test)
                #break

            print ('Epoch [{}/{}], Loss: {}'.format(epoch+1, 100, loss.data))
            #print(list(model.parameters()))
            print("-----------------------------------------------------------------------------")
            losses.append(loss.data)
        return model,loss_func,total_loss,losses

    def vol_test(self,model,criterion,test_dataloader,min_val,max_val):
        test_loss = 0
        accuracy = 0
        all_predictions = []
        for (examples, labels) in test_dataloader:
            output = model.forward(examples.float())
            un_normed_outputs = self.un_normalize(output, min_val,max_val)
            all_predictions.append(un_normed_outputs.detach())
            loss = criterion(output, labels.float()).item()
            test_loss += loss
        return test_loss, accuracy, all_predictions

    def vol_prediction_visualization(self,predictions,actual,min_val,max_val):
        for idx,item in enumerate(predictions):
              predictions[idx] = np.asscalar(item.numpy())
        print(predictions)
        actual = self.un_normalize(actual,min_val,max_val,typelist=True)
        print(actual)
        plt.figure(figsize=(20,10))
        plt.plot(np.array(actual),color='g')
        plt.plot(np.array(predictions),color='#FFA500')



def main():
    
    trainer = Trainer()

    # -- Preprocessing -- #
    raw_price_data = trainer.fetch_latest_BTC_JSON()
    data_df = trainer.parse_alphaV_JSON(raw_data=raw_price_data)
    #data_df = data_df.iloc[::-1] # Flip data

    # Seperating the y-data
    prices = np.array(data_df['4a. close (USD)'].tolist())
   
    # Temporary dataframe for creating an extra normalizer for re-scaling inference values later
    data_df_temp = data_df.drop(labels=['1a. open (USD)','1b. open (USD)','2b. high (USD)','3b. low (USD)','4a. close (USD)','4b. close (USD)','6. market cap (USD)'],axis=1) # ,'2a. high (USD)','3a. low (USD)'
    minmax_2 = preprocessing.MinMaxScaler()
    data_df_temp = pd.DataFrame(minmax_2.fit_transform(data_df_temp), columns=data_df_temp.columns)
    
    # -- Normalize the Data --
    min_max_scaler = preprocessing.MinMaxScaler()
    data_df = pd.DataFrame(min_max_scaler.fit_transform(data_df), columns=data_df.columns)
    data_df = trainer.table_edit(data_df)
    y_train = np.array(data_df['4a. close (USD)'].tolist())
    data_df = data_df.drop(labels=['4a. close (USD)'],axis=1)
    
    data_df = data_df.drop(labels=['1a. open (USD)','6. market cap (USD)'],axis=1)
 
    model = TimeRNN(bat_size=config['batch_size'],
                    in_features=3,
                    h_size=config['lstm_hidden_size'],
                    layer_amnt=config['lstm_num_layers']
                    ) 
    #model = TimeCNN()
    #model = MLP(3)

    losses, test_data_loader, loss_func, model, min_price, max_price, test_prices = trainer.train(model=model, 
                                                                                                  x_data=data_df.values,
                                                                                                  y_data=y_train,
                                                                                                  original_prices=prices,
                                                                                                  epochs=config['epochs']
                                                                                                  )
    #trainer.loss_visualize(losses)
    _, _, all_unnormed_outputs = trainer.validation_test(test_dataloader=test_data_loader,
                                                         criterion=loss_func, 
                                                         model=model, 
                                                         norm_min=min_price, 
                                                         norm_max=max_price)
    
    #trainer.prediction_visualization(minimum_price=min_price,maximum_price=max_price,close_prices=test_prices,model_predictions=all_unnormed_outputs)
    return minmax_2,model,min_price,max_price


if __name__ == '__main__':
    min_max_scaler,price_model,min_price,max_price = main()
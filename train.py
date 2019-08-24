import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import urllib3
import requests
import os
import yaml
import argparse
import json
import pickle
import pandas as pd
import cryptocompare
import time
from datetime import datetime
from utils.preprocessing import DataPrepper
from utils.preprocessing import train_test_split
#from models.architectures import MLP
#from models.architectures import TestRNN
from models.architectures import TimeCNN
#from models.architectures import TimeRNN

parser = argparse.ArgumentParser(description='Training Parameter Setter')
parser.add_argument('--tensorboard',
                    dest='tensorboard',
                    action='store_true',
                    help='saves tensorboard logs for debug and learning visualization')
parser.add_argument('--tcnn',
                    dest='tcnn',
                    action='store_true',
                    help='trains a temporal CNN model')
parser.add_argument('--save-model',
                    dest='save',
                    action='store_true',
                    help='saves the model as a .pt file')
parser.add_argument('--state-dict',
                    dest='state_dict_pt',
                    action='store_true',
                    help='saves model as a pytorch state dictionary .pt file')
parser.add_argument('--full-model',
                    dest='full_model_pt',
                    action='store_true',
                    help='saves the full model as a .pt file')
parser.add_argument('--onnx', 
                    dest='onnx', 
                    action='store_true',
                    help='saves the model as a .onnx file')
parser.add_argument('--output-dir', 
                    type=str,
                    default='outputs',
                    help='saves model in a given location')
args = parser.parse_args()
writer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TimeRNN(nn.Module):
    def __init__(self,bat_size,in_features,h_size,layer_amnt):
        super(TimeRNN,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.batch_sz = bat_size
        self.in_features = in_features
        self.h_size = h_size
        self.layer_amnt = layer_amnt
        
        self.lstm1 = nn.LSTM(input_size=self.in_features,
                             hidden_size=self.h_size,
                             num_layers=self.layer_amnt,
                             bias=True,
                             batch_first=False,
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
        h0 = hidden_init[0].to(self.device)
        c0 = hidden_init[1].to(self.device)
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

if(args.tensorboard):
    writer = SummaryWriter(log_dir=config['tensorboard_log_loc'])

class Trainer(object):
    def __init__(self,DataPrepper=None):
        self.x_data,self.y_data = DataPrepper.get_data()
        self.x_train,self.x_test,self.y_train,self.y_test = self.data_split(x_data=self.x_data,
                                                                            y_data=self.y_data)
        self.train_dataloader, self.test_dataloader = self.create_dataloaders(self.x_data,self.y_data)
    
    def create_dataloaders(self,x_data,y_data):
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
        return train_data_loader, test_data_loader

    def data_split(self,x_data,y_data):
        x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.2)#train_test_split(x_data,y_data,test_size=0.2,random_state=100,shuffle=False)
        return x_train, x_test, y_train, y_test

    def train(self,model, train_data, original_prices,epochs):
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

        print('-- Model Architecture --')
        print(model)

        # -- Model to CUDA GPU --
        if( str(device) == 'cuda'):
            print('Sending model to',torch.cuda.get_device_name(device),' GPU')
            #model = model.cuda()
            model.to(device)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        total_loss = 0
        losses = []
        for epoch in (range(epochs)):
            for i, (examples,labels) in tqdm(enumerate(train_data)):

                if( str(device) == 'cuda'):
                    examples = examples.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                y_predictions = model(examples.float())
                loss = loss_func(y_predictions.float(),labels.view(1,1).float())
                
                if(args.tensorboard):
                    writer.add_scalar(tag='loss',scalar_value=loss.data,global_step=i)

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
        return losses, loss_func, model, min_price, max_price

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
            loss = criterion(output, labels.view(1,1)).item()
            test_loss += loss

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

def get_model_name():
    minute = time.localtime().tm_min
    hour = time.localtime().tm_hour
    day = time.localtime().tm_mday
    month = time.localtime().tm_mon
    year = time.localtime().tm_year
    name = str(year) + "_" + str(month) + "_" + str(day) + "_"+ str(hour) + "_" + str(minute) + "_price_predictor"
    return name

def main():
    
    prepper = DataPrepper()
    trainer = Trainer(DataPrepper=prepper)


    if(args.tcnn):
        model = TimeCNN()
    else:
        model = TimeRNN(bat_size=config['batch_size'],
                        in_features=3,
                        h_size=config['lstm_hidden_size'],
                        layer_amnt=config['lstm_num_layers']
                        ) 
    #model = TimeCNN()
    #model = MLP(3)

    losses, loss_func, model, min_price, max_price = trainer.train(model=model, 
                                                                   train_data=trainer.train_dataloader,
                                                                   original_prices=prepper.prices,
                                                                   epochs=config['epochs']
                                                                  )
    #trainer.loss_visualize(losses)
    _, _, all_unnormed_outputs = trainer.validation_test(test_dataloader=trainer.test_dataloader,
                                                         criterion=loss_func, 
                                                         model=model, 
                                                         norm_min=min_price, 
                                                         norm_max=max_price)
    
    #trainer.prediction_visualization(minimum_price=min_price,maximum_price=max_price,close_prices=y_test,model_predictions=all_unnormed_outputs)
    
    return prepper.minmax_2,model,min_price,max_price,trainer.y_test,all_unnormed_outputs


if __name__ == '__main__':
    
    min_max_scaler,price_model,min_price,max_price,norm_test_vals,predictions = main()

    # -- Save Training Session Data (Test & Predictions) --
    for idx,val in enumerate(predictions):
        predictions[idx] = val.numpy().item()
    predictions = np.array(predictions)
    np.save('utils/test_data.npy',norm_test_vals)
    np.save('utils/predictions.npy',predictions)
    model_name = get_model_name()
    if(args.save == True):
        if(args.state_dict_pt == True):
            print('-- Saving Torch Model State Dictionary --')
            torch.save(price_model.state_dict(),'models/'+model_name+".pt")
        elif(args.full_model_pt == True):
            print('-- Saving Torch Model --')
            torch.save(price_model,'models/'+model_name+".pt")
        elif(args.onnx == True):
            print('-- Exporting to ONNX --')
            dummy_input = torch.tensor([[1, 2, 3]]).float()
            model_path = os.path.join(args.output_dir, model_name+".onnx")
            torch.onnx.export(price_model, dummy_input, model_path)

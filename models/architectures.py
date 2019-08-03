import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        h0 = hidden_init[0].to(device)
        c0 = hidden_init[1].to(device)
        x,hidden = self.lstm1( x,(h0,c0))
        x = F.leaky_relu(self.fc1(x[-1].view(self.batch_sz,-1)))
        return x
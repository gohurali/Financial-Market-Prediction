import torch
import torch.nn as nn
import torch.nn.functional as F

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
"""
class TestRNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_amnt):
        super(TestRNN,self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_amnt = layer_amnt
        
        self.lstm1 = nn.LSTM(input_size=self.input_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=1,
                             bias=True,
                             batch_first=False,
                             dropout=0,
                             bidirectional=False
                            )
        self.fc1 = nn.Linear(in_features=150,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=32)
        self.fc4 = nn.Linear(in_features=32,out_features=1)
        
        pass
    def forward(self,x):
        pass
      
"""
Temporal Convolutional Neural Network (CNN)
Time series based convolutional neural network
"""
class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=256,kernel_size=2,stride=1,padding=0)
        self.mp1 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.conv2 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=0)
        self.mp2 = nn.MaxPool1d(kernel_size=2,stride=1)
        self.fc1 = nn.Linear(in_features=512,out_features=512)
        self.fc2 = nn.Linear(in_features=512,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=1)
    def forward(self,x):
        x = x.unsqueeze(0)
        x = F.leaky_relu(self.conv1(x))
        #x = self.mp1(x)
        #x = F.relu(self.conv2(x))
        #x = self.mp2(x)
        x = x.reshape(-1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils as ut
from torch.autograd import Variable
import numpy as np

class MLP(nn.Module):
    def __init__(self,dim_in,dim_out,num_hidden_layer=3,width=100):
        super().__init__()
        lst_layer=list()
        lst_layer.append(nn.Linear(dim_in,width))
        lst_layer.append(nn.ReLU())
        for i in range(0,num_hidden_layer-1):
            lst_layer.append(nn.Linear(width,width))
            lst_layer.append(nn.ReLU())
        lst_layer.append(nn.Linear(width,dim_out))
        self.net=nn.Sequential(*lst_layer)
    def forward(self,x):
        return self.net(x)

class Encoder(nn.Module):
    '''
	-architecture similar to a PointNet model
    -P(z|x,Phi) is modeled as a gaussian
    -return mean (dim z) and variance (dim z) of the gaussian
    '''
    def __init__(self, dim_rep,dim_z=3):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2*dim_z)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m,v
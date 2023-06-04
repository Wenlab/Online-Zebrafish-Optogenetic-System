import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torchsummary import summary

import hiddenlayer as h


class AffineNetwork(nn.Module):
    def __init__(self,device):
        super(AffineNetwork,self).__init__()
        self.device=device

        
        self.feature_extractor1 = Feature_Extractor(self.device)
        self.feature_extractor2 = Feature_Extractor(self.device)
        
        self.feature_combiner=nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regression_network = Regression_Network()


    def forward(self,fix,moving):
        x1 = self.feature_extractor1(fix)
        x2 = self.feature_extractor2(moving)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.feature_combiner(x)
        
        x = x.view(-1,256)
        x = self.regression_network(x)
        return x

class Forward_Layer(nn.Module):
    def __init__(self, channels, pool=False):
        super(Forward_Layer, self).__init__()
        self.pool = pool
        if self.pool:
            self.pool_layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3)
            )
            self.layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3),
                nn.GroupNorm(8, 2*channels),
                nn.PReLU(),
                nn.Conv2d(2*channels, 2*channels, 3, stride=1, padding=1),
                nn.GroupNorm(8, 2*channels),
                nn.PReLU(),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(8, channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(8, channels),
                nn.PReLU(),
            )

    def forward(self, x):
        if self.pool:
            return self.pool_layer(x) + self.layer(x)
        else:
            return x + self.layer(x)


class Feature_Extractor(nn.Module):
    def __init__(self, device):
        super(Feature_Extractor,self).__init__()
        self.device=device
        
        self.input_layer=nn.Sequential(

            nn.Conv2d(52, 64, 3)
        )
        self.layer_1=Forward_Layer(64,pool=True)
        self.layer_2=Forward_Layer(128,pool=False)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256, pool=False)
        self.layer_5 = Forward_Layer(256, pool=True)
        self.layer_6 = Forward_Layer(512, pool=False)
        self.last_layer = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self,x):
        x=self.input_layer(x)
        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.layer_3(x)
        x=self.layer_4(x)
        return x


class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,12)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

 
#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()
 


def load_network(device, path=None):
    model = AffineNetwork(device)
    model = model.to(device)

    if path is not None:
        model = torch.load(path)
        model.eval()
    return model


def test_forward_pass():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    
    y_size = 77
    x_size = 95
    no_channels = 52

    batch_size = 4
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    result = model(example_source, example_target)
    print(result.shape)
    print(result)

def run():
    test_forward_pass()

if __name__ == "__main__":
    run()
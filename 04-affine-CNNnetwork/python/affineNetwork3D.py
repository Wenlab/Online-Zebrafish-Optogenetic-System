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
            nn.Conv3d(512, 256, (3,3,3), stride=(3,3,3), padding=3),
            nn.GroupNorm(256, 256),
            # nn.InstanceNorm2d(512, 512),
            nn.PReLU(),
            nn.Conv3d(256, 128,(3,3,3), stride=(3,3,3), padding=3),
            nn.GroupNorm(128, 128),
            nn.PReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.regression_network = Regression_Network()


    def forward(self,fix,moving):
        x1 = self.feature_extractor1(fix)
        x2 = self.feature_extractor2(moving)
        
        x = torch.cat((x1, x2), dim=1)
        # print(x.shape)
        x = self.feature_combiner(x)
        # print(x.shape)
        x = x.view(-1,128)
        x = self.regression_network(x)
        return x

class Forward_Layer(nn.Module):
    def __init__(self, channels, pool=False):
        super(Forward_Layer, self).__init__()
        self.pool=pool
        if self.pool:
            self.pool_layer = nn.Sequential(
                nn.Conv3d(channels, 2*channels, (3,3,3), stride=2, padding=3)
            )
            self.layer=nn.Sequential(
                nn.Conv3d(channels, 2*channels, (3,3,3), stride=2, padding=3),
                nn.GroupNorm(2*channels, 2*channels), 
                nn.PReLU(),
                # nn.Conv3d(2*channels, 2*channels, (3,3,3), stride=1, padding=1),
                # nn.GroupNorm(8, 2*channels),
                # nn.PReLU(),
            )
        else:
            self.layer=nn.Sequential(
                nn.Conv3d(channels, channels, (3,3,3), stride=1, padding=1),
                nn.GroupNorm(channels, channels), 
                nn.PReLU(),
                # nn.Conv3d(channels, channels, (3,3,3), stride=1, padding=1),
                # nn.GroupNorm(8, channels),
                # nn.PReLU(),
            )
    def forward(self, x):
        if self.pool:
            # print(self.pool_layer(x).shape)
            # print(self.layer(x).shape)
            return self.pool_layer(x) + self.layer(x)
        else:
            
            return x + self.layer(x)

class Feature_Extractor(nn.Module):
    def __init__(self, device):
        super(Feature_Extractor,self).__init__()
        self.device=device
        self.input_layer=nn.Sequential(

            nn.Conv3d(1, 64, (3,3,3), stride=(3,3,3), padding=3),
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
        # x=self.layer_5(x)
        # x=self.layer_6(x)
        # x=self.last_layer(x)
        # print(x.size())
        return x

class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,12)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

def load_network(device, path=None):
    model = AffineNetwork(device)
    # model.apply(weigth_init)
    model = model.to(device)

    if path is not None:
        # model.load_state_dict(torch.load(path))
        model = torch.load(path)
        model.eval()
    return model


def test_forward_pass():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    
    y_size = 77
    x_size = 95
    depth = 52
    channel=1

    test_random=torch.rand((channel,y_size,x_size))
    test_random=test_random.unsqueeze(dim=1) 
    print(test_random.shape)


    batch_size = 4
    example_source = torch.rand((batch_size,channel, depth, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size,channel,depth, y_size, x_size)).to(device)

    result = model(example_source, example_target)
    print(result.shape)
    print(result)

def run():
    test_forward_pass()

if __name__ == "__main__":
    run()
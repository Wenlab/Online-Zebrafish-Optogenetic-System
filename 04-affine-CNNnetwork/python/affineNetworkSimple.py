import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from torchvision import models

import torchvision.models as models

class Affine_Network(nn.Module):
    def __init__(self, device):
        super(Affine_Network, self).__init__()
        self.device = device

        self.feature_extractor = Feature_Extractor(self.device)
        self.regression_network = Regression_Network()

        self.resnet1=models.resnet18(pretrained=False)
        self.resnet1.conv1= nn.Conv2d(52, 64, kernel_size=7, stride=2, padding=3,bias=False)
        num_ftrs=self.resnet1.fc.in_features
        self.resnet1.fc=nn.Linear(num_ftrs,128)


        self.resnet2=models.resnet18(pretrained=False)
        self.resnet2.conv1= nn.Conv2d(52, 64, kernel_size=7, stride=2, padding=3,bias=False)
        num_ftrs=self.resnet2.fc.in_features
        self.resnet2.fc=nn.Linear(num_ftrs,128)
        # self.sigmoid=torch.nn.Sigmoid()

    def forward(self, source, target):
        x = self.feature_extractor(source-target)
        # x1=self.resnet1(source)
        # x2=self.resnet2(target)
        # x=torch.cat((x1,x2),dim=1)
        x = x.view(-1, 256)
        x = self.regression_network(x)
        # x = F.softmax(x, dim=1)
        # x=self.resnet(torch.cat((source, target), dim=1))
        # x=self.resnet(source)
        # x=self.sigmoid(x)
        # x=self.resnet(source-target)

        # regress a bias
        # temp = torch.tensor([1.0,0,0,1,0,0])
        # adjust = temp.repeat(x.shape[0],1)
        # adjust = adjust.to(self.device)
        # x = 0.01*x + adjust
        return x

class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 12),
        )

    def forward(self, x):
        x = self.fc(x)
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
        super(Feature_Extractor, self).__init__()
        self.device = device
        self.input_layer = nn.Sequential(
            # add norm??
            nn.Conv2d(52, 64, 3, stride=2, padding=3),
        )
        self.layer_1 = Forward_Layer(64, pool=True)
        self.layer_2 = Forward_Layer(128, pool=False)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256, pool=False)
        self.layer_5 = Forward_Layer(256, pool=True)
        self.layer_6 = Forward_Layer(512, pool=True)

        self.last_layer = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )



    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.last_layer(x)
        return x

def load_network(device, path=None):
    model = Affine_Network(device)
    if path is not None:
        print("load model from: ",path)
        model.load_state_dict(torch.load(path))
        model.eval()
    model = model.to(device)

    return model



def test_forward_pass():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = load_network(device)

    y_size = 77
    x_size = 95
    # y_size = 224
    # x_size = 224
    no_channels = 52
    # summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])
    # print(model)

    batch_size = 16
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    # t=torch.cat((example_source, example_target), dim=1)
    # print(t.shape)

    result = model(example_source, example_target)
    print(result.shape)

def run():
    test_forward_pass()

if __name__ == "__main__":
    run()
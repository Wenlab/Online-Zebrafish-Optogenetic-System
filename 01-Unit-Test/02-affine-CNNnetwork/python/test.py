import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import nibabel as nib

import time

import skimage.io
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F

from sitkAffine import warpTensors,saveResultTensor

import affineNetWorkModel as affineNet

import dataLoader as dl

import os


device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


modelPath='affineModel.pt'
model=affineNet.load_network(device, path=modelPath)
model=model.to(device)



loss_function=torch.nn.L1Loss(reduction='mean')

dimension=3
testPath="../test/"
savePath="../NetResult/"
if not os.path.exists(savePath):
    os.makedirs(savePath)


test_loader=dl.AffineRegistrationDataset(testPath)
batchSize=1
test_dataloader = torch.utils.data.DataLoader(test_loader, batch_size = batchSize, shuffle = False, num_workers = 1)

testSize=len(test_dataloader.dataset)
print("test size: ", testSize)
testLoss=0
for i, data in enumerate(test_dataloader):
    print(i)
    movings, fixes ,affineMatrices = data[0].to(device), data[1].to(device),data[2].to(device)
    matrix_caculate=model(movings,fixes)
    loss=loss_function(matrix_caculate,affineMatrices)
    testLoss += loss.item()

    result=warpTensors(movings,matrix_caculate,dimension)
    saveResultTensor(result,savePath,i)


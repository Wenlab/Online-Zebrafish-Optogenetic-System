import SimpleITK as sitk
import torch
import numpy as np

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F


from dataLoader import normalize
from sitkAffine import warpTensors,reAffineMarix,saveResultTensor

# device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
fixImgPath='/home/customer/kexin/pytorchTest/testAffine/fix.nii'
movingImgPath='/home/customer/kexin/pytorchTest/testAffine/moving.nii'

device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

fixImg=sitk.ReadImage(fixImgPath)
movingImg=sitk.ReadImage(movingImgPath)
print(fixImg.GetSize())

fixImg=sitk.GetArrayFromImage(fixImg)
movingImg=sitk.GetArrayFromImage(movingImg)

print(fixImg.shape)

print(np.max(fixImg))
print(np.max(movingImg))

movingImg = normalize(movingImg)
fixImg = normalize(fixImg)

movingImg_tensor, fixImg_tensor = torch.from_numpy(movingImg.astype(np.float32)), torch.from_numpy(fixImg.astype(np.float32))

print(fixImg_tensor[:,35,25])


movingImg_tensor=movingImg_tensor.unsqueeze(0).to(device)
fixImg_tensor=fixImg_tensor.unsqueeze(0).to(device)

print(torch.max(movingImg_tensor))
print(torch.max(fixImg_tensor))


modelPath='affineNetScript_cpu.pt'
model = torch.load(modelPath)
model.eval().to(device)

# y_size = 95
# x_size = 77
# no_channels = 52

# batch_size = 1
# example_source = torch.ones((batch_size, no_channels, y_size, x_size)).to(device)
# example_target = torch.ones((batch_size, no_channels, y_size, x_size)).to(device)


# print(movingImg_tensor.shape)


for j in range(100):
    T1 = time.clock()
    for i in range(1000):
        matrix_caculate=model(movingImg_tensor, fixImg_tensor)
    T2 =time.clock()
    print('模型运行1000次:%s毫秒' % ((T2 - T1)*1000))
    print('平均时间:%s毫秒' % ((T2 - T1)))
# print(matrix_caculate)
# result=warpTensors(movingImg_tensor,matrix_caculate,3)
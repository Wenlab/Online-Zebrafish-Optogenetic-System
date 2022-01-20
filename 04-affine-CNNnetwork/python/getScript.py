import SimpleITK as sitk
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.nn.functional as F

from dataLoader import normalize
from sitkAffine import warpTensors,reAffineMarix,saveResultTensor
import affineNetWorkModel as affineNet

# device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
fixImgPath='/home/customer/kexin/pytorchTest/testAffine/fix.nii'
movingImgPath='/home/customer/kexin/pytorchTest/testAffine/moving.nii'

device=torch.device("cpu")

fixImg=sitk.ReadImage(fixImgPath)
movingImg=sitk.ReadImage(movingImgPath)

fixImg=sitk.GetArrayFromImage(fixImg)
movingImg=sitk.GetArrayFromImage(movingImg)

print(np.max(fixImg))
print(np.max(movingImg))

movingImg = normalize(movingImg)
fixImg = normalize(fixImg)

movingImg_tensor, fixImg_tensor = torch.from_numpy(movingImg.astype(np.float32)), torch.from_numpy(fixImg.astype(np.float32))
movingImg_tensor=movingImg_tensor.unsqueeze(0)
fixImg_tensor=fixImg_tensor.unsqueeze(0)

print(torch.max(movingImg_tensor))
print(torch.max(fixImg_tensor))


modelPath='/home/customer/kexin/pytorchTest/model/affineNet_0105_0824_0924_CMTK_AM_TM3_x10.pt'
model=affineNet.load_network(device, path=modelPath)
model=model.to(device)

# y_size = 77
# x_size = 95
# no_channels = 52

# batch_size = 1
# example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
# example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

print(movingImg_tensor.shape)
matrix_caculate=model(movingImg_tensor,fixImg_tensor)
print(matrix_caculate)
result=warpTensors(movingImg_tensor,matrix_caculate,3)

saveResultTensor(result,'/home/customer/kexin/pytorchTest/testAffine/',1)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (movingImg_tensor, fixImg_tensor))

traced_script_module.save("affineNetScript_TM_cpu.pt")
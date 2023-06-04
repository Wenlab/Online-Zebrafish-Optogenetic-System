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

fixImgPath='../testAffine/fix.nii'
movingImgPath='../testAffine/moving.nii'

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


modelPath='affineModel.pt'
model=affineNet.load_network(device, path=modelPath)
model=model.to(device)


print(movingImg_tensor.shape)
matrix_caculate=model(movingImg_tensor,fixImg_tensor)
print(matrix_caculate)
result=warpTensors(movingImg_tensor,matrix_caculate,3)

saveResultTensor(result,'../testAffine/',1)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (movingImg_tensor, fixImg_tensor))

traced_script_module.save("affineNetScript.pt")
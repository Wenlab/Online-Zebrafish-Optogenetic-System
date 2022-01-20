
import torch
from torch.nn import functional as F
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def ncc_global(sources,targets,device="cpu",**params):
    size=sources.size(2)*sources.size(3)*sources.size(1)
    sources_mean=torch.mean(sources,dim=(1,2,3)).view(sources.size(0),1,1,1)
    targets_mean=torch.mean(targets,dim=(1,2,3)).view(targets.size(0),1,1,1)
    sources_std=torch.std(sources,dim=(1,2,3)).view(sources.size(0),1,1,1)
    targets_std=torch.std(targets,dim=(1,2,3)).view(targets.size(0),1,1,1)

    ncc=(1/size)*torch.sum((sources-sources_mean)*(targets-targets_mean)/(sources_std*targets_std),dim=(1,2,3))

    return ncc

def ncc_losses_global(sources, targets, device="cpu", **params):
    ncc = ncc_global(sources, targets, device=device, **params)
    ncc = torch.mean(ncc)
    if ncc != ncc:
        return torch.autograd.Variable(torch.Tensor([1]), requires_grad=True).to(device)
    return ncc


def normalize(image):
    if(np.all(image == 0)):
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))

if __name__ == "__main__":
    movingPath="/home/customer/kexin/pytorchTest/allDataset/0824CMTK_AM/train/movingImg/r20210824_X7_0002.nii"
    fixPath="/home/customer/kexin/pytorchTest/Obj_ref.nii"

    movingImg=sitk.GetArrayFromImage(sitk.ReadImage(movingPath))
    fixImg=sitk.GetArrayFromImage(sitk.ReadImage(fixPath))

    # normalize to 0~1
    movingImg = normalize(movingImg)
    fixImg = normalize(fixImg)

    movingImg_tensor, fixImg_tensor = torch.from_numpy(movingImg.astype(np.float32)), torch.from_numpy(fixImg.astype(np.float32))

    
    movingImg_tensor=movingImg_tensor.unsqueeze(dim=0)   
    fixImg_tensor=fixImg_tensor.unsqueeze(dim=0)
    # if test 2D image
    # movingImg_tensor=movingImg_tensor.unsqueeze(dim=1)   
    # fixImg_tensor=fixImg_tensor.unsqueeze(dim=1)


    # batch_size = 8
    # y_size = 77
    # x_size = 95
    # no_channels = 52

    # example_source = torch.rand((batch_size, no_channels, y_size, x_size))
    # example_target = torch.rand((batch_size, no_channels, y_size, x_size))

    print(movingImg_tensor.shape)
    print(fixImg_tensor.shape)
    ncc=ncc_losses_global(movingImg_tensor,fixImg_tensor,device="cpu")

    print(ncc)


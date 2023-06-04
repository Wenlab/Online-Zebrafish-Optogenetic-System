import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import random
import SimpleITK as sitk
import pandas as pd

import torch
import torch.utils as utils

from skimage import transform

from critical import ncc_losses_global

from sitkAffine import saveResultTensor, warpTensors


def collate_to_list_Affine(batch):
    moving = [item[0].view(item[0].size(0), item[0].size(1), item[0].size(2)) for item in batch]
    fix = [item[1].view(item[1].size(0), item[1].size(1), item[0].size(2)) for item in batch]
    matrix=[item[2].view(item[2].size(0)) for item in batch]
    return moving, fix, matrix


class AffineRegistrationDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,transforms=None):
        self.data_path=data_path
        self.movingImg_path=os.path.join(data_path,"movingImg")
        self.fixImg_path=os.path.join(data_path,"fixImg")
        self.affineMatrices_path=os.path.join(data_path,"affineMatrices.csv")

        self.all_movingImgName=os.listdir(self.movingImg_path)
        self.all_movingImgName.sort()

        self.all_fixImgName=os.listdir(self.fixImg_path)
        self.all_fixImgName.sort()

        self.affineMatrices=pd.read_csv(self.affineMatrices_path,header=None)

        self.transforms=transforms


    def __len__(self):
        return len(self.all_movingImgName)


    def __getitem__(self,idx):
        movingImg_id=self.all_movingImgName[idx]
        fixImg_id=self.all_fixImgName[idx]

        movingImg_path=os.path.join(self.movingImg_path,movingImg_id)
        fixImg_path=os.path.join(self.fixImg_path,fixImg_id)

        movingImg=sitk.GetArrayFromImage(sitk.ReadImage(movingImg_path))
        fixImg=sitk.GetArrayFromImage(sitk.ReadImage(fixImg_path))
        
        if self.transforms is not None:
            movingImg, fixImg, _ = self.transforms(movingImg, fixImg)
   

        affineMatrix=self.affineMatrices.values[idx,:]

        # normalize to 0~1
        movingImg = normalize(movingImg)
        fixImg = normalize(fixImg)

        movingImg_tensor, fixImg_tensor = torch.from_numpy(movingImg.astype(np.float32)), torch.from_numpy(fixImg.astype(np.float32))

        affineMatrix=torch.from_numpy(affineMatrix.astype(np.float32))

        return movingImg_tensor,fixImg_tensor,affineMatrix


def normalize(image):
    if(np.all(image == 0)):
        return image
    else:
        return (image - np.min(image)) / (np.max(image) - np.min(image))



def show_batch_images(sample_batch,batchSize,filename):
    movingImg=sample_batch[0]
    fixImg=sample_batch[1]
    label=sample_batch[2]

    savePath='/home/customer/kexin/pytorchTest/testDataloader/'

    for i in range(batchSize):
        label_ = label[i].item()
        image_=movingImg[i].squeeze(dim=0)
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.savefig(os.path.join(savePath, str(filename) + ".png"))
    plt.close()

if __name__ == "__main__":
    trainingPath="/home/customer/kexin/pytorchTest/allDataset/0824CMTK_AM_TM3_checkAffineResult/val/"
    savepath="/home/customer/kexin/pytorchTest/allDataset/0824CMTK_AM_TM3_checkAffineResult/val/result/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    training_loader=AffineRegistrationDataset(trainingPath)
    training_dataloader=torch.utils.data.DataLoader(training_loader, batch_size = 1, shuffle = False, num_workers = 4)
    filename=0
    n1=0
    n2=0
    for i, data in enumerate(training_dataloader):
        movings, fixes ,affineMatrices = data[0],data[1],data[2]
        print(movings.shape)
        result=warpTensors(movings,affineMatrices,3)
        saveResultTensor(result,savepath,i)
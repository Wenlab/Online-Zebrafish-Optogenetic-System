import SimpleITK as sitk
import numpy as np

import torch
from torch.nn import functional as F
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def reAffineMarix(affineMatrix,dimension):
    if dimension==3:
        affineMatrix=affineMatrix/100
        affineMatrix[0]=affineMatrix[0]/10+1
        affineMatrix[4]=affineMatrix[4]/10+1
        affineMatrix[8]=affineMatrix[8]/10+1
        affineMatrix[9]=affineMatrix[9]*77
        affineMatrix[10]=affineMatrix[10]*95
        affineMatrix[11]=affineMatrix[11]*52
    if dimension==2:
        affineMatrix[0]=affineMatrix[0]/100+1
        affineMatrix[1]=affineMatrix[1]/10
        affineMatrix[2]=affineMatrix[2]/10
        affineMatrix[3]=affineMatrix[3]/100+1

    # print(affineMatrix)
    return affineMatrix


def resample(image, transform):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = image
    interpolator = sitk.sitkCosineWindowedSinc
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def warpImg(image,affineMatrix,dimension):

    translation=sitk.TranslationTransform(dimension)
    translation.SetOffset(affineMatrix[-dimension:])

    affine=sitk.AffineTransform(dimension)
    affine.SetMatrix(affineMatrix[0:dimension*dimension])

    composite=sitk.CompositeTransform(dimension)
    composite.AddTransform(translation)
    composite.AddTransform(affine)

    resampled=resample(image,composite)
    return resampled


def warpTensors(movingImgs,affineMatrices,dimension):

    result=torch.zeros(movingImgs.shape)
    for i in range(movingImgs.size(0)):
        movImg=movingImgs[i,:,:,:]
        am=affineMatrices[i,:]
        movImg = torch.squeeze(movImg, 0)
        am = torch.squeeze(am, 0)

        am=reAffineMarix(am,dimension)
        movImg=movImg.cpu().detach().numpy()
        am_n=am.cpu().detach().numpy().tolist()
        
        movImg = sitk.GetImageFromArray(movImg)

        warpResult=warpImg(movImg,am_n,dimension)

        warpResult=torch.from_numpy(sitk.GetArrayFromImage(warpResult).astype(np.float32))
        result[i,:,:,:]=warpResult

    return result

def saveResultTensor(result,savePath,batchNum):
    for i in range(result.size(0)):
        img=result[i,:,:,:]
        img = torch.squeeze(img, 0)
        img=img.numpy()
        img=sitk.GetImageFromArray(img)
        sitk.WriteImage(img,savePath+'%04d'%batchNum+'_'+'%04d'%i+'.nii')



if __name__ == "__main__":
    movingImgPath='../test/movingImg/r20210824_X1_0037.nii'
    dimension=3
    movingImg=sitk.ReadImage(movingImgPath)
    affineMatrix=[ 1.00539,	0.00619452,	-0.0314952,	-0.00878725,	0.993521,	-0.0116533,	0.0289862,	0.012498,	1.00583,	0.887444,	1.26929,	-2.42047]
    re=warpImg(movingImg,affineMatrix,dimension)

    sitk.WriteImage(re,'testAffine.nii')
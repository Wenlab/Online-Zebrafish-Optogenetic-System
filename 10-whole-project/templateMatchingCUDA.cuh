#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tchar.h>
#include <io.h>
#include <string>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include <FreeImage.h>
#include <npp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h> 
#include <thrust/sort.h> 
#include <thrust/copy.h> 
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <chrono>//标准模板库中与时间有关的头文件
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>
#include "gdal_mdreader.h"
#include "gdalwarper.h"
#include "ogrsf_frmts.h"




//npp的图像旋转函数nppiRotate_32f_C1R使用方法：
//若旋转角度为正，图像从左下角行优先开始读的：顺时针旋转。左上角开始读的：逆时针旋转
//若旋转角度为负，图像从左下角行优先开始读的：逆时针旋转。左上角开始读的：顺时针旋转
float *ObjRecon_imrotate3(float *ObjRecon, double nAngle);

//按照X轴旋转
float *ObjRecon_imrotate3_X(float *imageRotated3D, double nAngle);

//重采样到指定长宽
int reSampleGDAL(const char* pszSrcFile, const char* pszOutFile, int newWidth, int newHeight, GDALResampleAlg eResample = GRA_Bilinear);

//重采样到指定长宽
int reSampleGDAL_1(float *ArrayBand, int width, int height, int nBandCount, GDALDataType dataType,
	float *ArrayBand_out, int newWidth, int newHeight, GDALResampleAlg eResample = GRA_Bilinear);

//取XY平面的MIP
__global__ void kernel_1(float *ObjRecon_gpu, int height, int width, float *image2D_XY_gpu);

//图像平均值作为阈值，对图像做二值化
__global__ void kernel_2(float *image2D_XY_gpu, int total, double image2D_XY_mean, float *img2DBW_XY_gpu);

__global__ void kernel_3(float *template_roXY_gpu, float *img2DBW_XY_gpu, int rotationAngleXY_size, double *err_XY_gpu);

//void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu);

__global__ void kernel_4(float *imageRotated3D_gpu, float *image2D_YZ_gpu);

__global__ void kernel_5(float *image2D_YZ_gpu, double image2D_YZ_mean, float *img2DBW_YZ_gpu);

__global__ void kernel_6(float *template_roYZ_gpu, float *img2DBW_YZ_gpu, int rotationAngleYZ_size, double *err_YZ_gpu);

//维度变换
__global__ void kernel_7(float *imageRotated3D_gpu, float *imageRotated3D_gpu_1);

//按照X轴旋转
void ObjRecon_imrotate3_X_gpu(float *imageRotated3D_gpu_1, double nAngle, float *imageRotated3D_gpu_2);

//再变换到原来的维度
__global__ void kernel_8(float *imageRotated3D_gpu_2, float *imageRotated3D_gpu);

__global__ void kernel_9(float *imageRotated3D_gpu, double imageRotated3D_x_mean, int *BWObjRecon_gpu);

__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2);
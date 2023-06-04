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
#include <chrono>
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>
#include "gdal_mdreader.h"
#include "gdalwarper.h"
#include "ogrsf_frmts.h"


//get the MIP of XY plane
__global__ void kernel_1(float *ObjRecon_gpu, int height, int width, float *image2D_XY_gpu);

//The average value of the image is used as a threshold, and the image is binarized
__global__ void kernel_2(float *image2D_XY_gpu, int total, double image2D_XY_mean, float *img2DBW_XY_gpu);

__global__ void kernel_3(float *template_roXY_gpu, float *img2DBW_XY_gpu, int rotationAngleXY_size, double *err_XY_gpu);

__global__ void kernel_4(float *imageRotated3D_gpu, float *image2D_YZ_gpu);

__global__ void kernel_5(float *image2D_YZ_gpu, double image2D_YZ_mean, float *img2DBW_YZ_gpu);

__global__ void kernel_6(float *template_roYZ_gpu, float *img2DBW_YZ_gpu, int rotationAngleYZ_size, double *err_YZ_gpu);

//Dimensional Transformation
__global__ void kernel_7(float *imageRotated3D_gpu, float *imageRotated3D_gpu_1);


//Transform again to the original dimension
__global__ void kernel_8(float *imageRotated3D_gpu_2, float *imageRotated3D_gpu);

__global__ void kernel_9(float *imageRotated3D_gpu, double imageRotated3D_x_mean, int *BWObjRecon_gpu);

__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2);

__global__ void kernel_11(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int Corner0, int Corner2);
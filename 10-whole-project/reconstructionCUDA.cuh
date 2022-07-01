#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex>
#include <vector>
#include <cufft.h>
#include <chrono>//标准模板库中与时间有关的头文件
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//int clockRate = 1.0;
//
//float scale = 1.0 / 4;
//int ItN = 10; // 迭代次数
//int ROISize = 100;// ROI 大小 不用管
//int BkgMean = 140;// 背景均值
//int SNR = 200;// SNR不用管
//int NxyExt = 0;//用来填充的范围，本程序是0，不用填充了
//int PSF_size_1 = 512;
//int PSF_size_2 = 512;
//int PSF_size_3 = 50;
//int Nxy = PSF_size_1 + NxyExt * 2; // 是个常数 可以直接存显存里
//int Nz = PSF_size_3; // 可以直接用常数 存显存里  Nz = 50
//
//int threadNum_123 = 256;
//int blockNum_123 = (PSF_size_1*PSF_size_2*PSF_size_3 - 1) / threadNum_123 + 1;
//int threadNum_12 = 256;
//int blockNum_12 = (PSF_size_1*PSF_size_2 - 1) / threadNum_12 + 1;
//int threadNum_ROI = 256;
//int blockNum_ROI = (ROISize * 2 * ROISize * 2 * Nz - 1) / threadNum_ROI + 1;
//dim3 block(8, 8, 8);
//dim3 grid((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, (PSF_size_3 + block.z - 1) / block.z);
//dim3 block_sum(32, 32, 1);
//dim3 grid_sum((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, 1);
//
//
//////*----1、使用cufftPlan2d的方法进行二维fft----------*/
//cufftHandle plan;
//cufftResult res = cufftPlan2d(&plan, PSF_size_1, PSF_size_2, CUFFT_C2C);



__global__ void Zhuan_Complex_kernel(float *PSF_1_gpu, cufftComplex *PSF_1_gpu_Complex, int total);

__global__ void PSF_unshort(float *PSF_1_gpu, unsigned short *PSF, int total);

__global__ void initial_kernel_1(float *ImgEst, float *Ratio, int total);

__global__ void gpuObjRecon_fuzhi(float *gpuObjRecon, int total);

__global__ void initial_kernel_3(float *gpuObjRecROI, int total);

__global__ void ImgExp_ge(unsigned short *Img_gpu, int BkgMean, float *ImgExp, int total);

__global__ void Ratio_fuzhi(float *Ratio, int total);

__global__ void OTF_mul_gpuObjRecon_Complex(cufftComplex *OTF, cufftComplex *gpuObjRecon_Complex, int total);

__global__ void ifftshift_real_max(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3);

__global__ void ifftshift(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3, cufftComplex *OTF_ifftshift);

__global__ void float_temp_sum(float *float_temp, float *ImgEst, int PSF_size_1, int PSF_size_2, int PSF_size_3);

__global__ void Ratio_fuzhi_2(float *ImgExp, float *ImgEst, float Tmp, int SNR, float *Ratio, int total);

__global__ void Ratio_Complex_ge(float *ImgExp, float *ImgEst, float Tmp, int SNR, cufftComplex *Ratio_Complex, int total);

__global__ void fftRatio_ge(cufftComplex *Ratio_Complex, cufftComplex *fftRatio, int PSF_size_1, int PSF_size_2, int PSF_size_3);

__global__ void fftceshi_gpu_fuzhi(cufftComplex *PSF_1_gpu_Complex, cufftComplex *fftceshi_gpu, int total);

__global__ void ifft2_divide(cufftComplex *OTF, int total, int scale);

__global__ void real_multiply(float *gpuObjRecon, float *float_temp, int total);

__global__ void fftRatio_mul_conjOTF(cufftComplex *fftRatio, cufftComplex *OTF, int total);

__global__ void cropReconImage_kernel(float *gpuObjRecon, float *gpuObjRecon_crop);



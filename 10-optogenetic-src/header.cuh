#ifndef HEADER_CUH+
#define HEADER_CUH

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
#include <chrono>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



int clockRate = 1.0;

float scale = 1.0 / 4;
int ItN = 10; 
int ROISize = 100;
int BkgMean = 140;
int SNR = 200;
int NxyExt = 0;
int PSF_size_1 = 512;
int PSF_size_2 = 512;
int PSF_size_3 = 50;
int Nxy = PSF_size_1 + NxyExt * 2; 
int Nz = PSF_size_3; 

int threadNum_123 = 256;
int blockNum_123 = (PSF_size_1*PSF_size_2*PSF_size_3 - 1) / threadNum_123 + 1;
int threadNum_12 = 256;
int blockNum_12 = (PSF_size_1*PSF_size_2 - 1) / threadNum_12 + 1;
int threadNum_ROI = 256;
int blockNum_ROI = (ROISize * 2 * ROISize * 2 * Nz - 1) / threadNum_ROI + 1;
dim3 block(8, 8, 8);
dim3 grid((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, (PSF_size_3 + block.z - 1) / block.z);
dim3 block_sum(32, 32, 1);
dim3 grid_sum((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, 1);


cufftHandle plan;
cufftResult res = cufftPlan2d(&plan, PSF_size_1, PSF_size_2, CUFFT_C2C);




#endif
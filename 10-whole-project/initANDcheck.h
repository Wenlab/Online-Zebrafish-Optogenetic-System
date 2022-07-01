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
//#include <cufft.h>

#include <assert.h>
//#include <chrono>//标准模板库中与时间有关的头文件

using std::string;


//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop);
//CUDA 初始化
bool InitCUDA();
void check(cudaError_t res, string warningstring);
//用法check1(cudaMalloc((void**)&aa, 10 * sizeof(float)), "aa cudaMalloc Error", __FILE__, __LINE__);
void check1(cudaError_t res, string warningstring, const char *file, int linenum);
//查看GPU运行是否正确
void checkGPUStatus(cudaError_t cudaGetLastError, string warningstring);


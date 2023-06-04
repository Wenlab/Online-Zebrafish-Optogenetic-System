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

#include <assert.h>

using std::string;


//Print device information
void printDeviceProp(const cudaDeviceProp &prop);
bool InitCUDA();
void check(cudaError_t res, string warningstring);
//example: check1(cudaMalloc((void**)&aa, 10 * sizeof(float)), "aa cudaMalloc Error", __FILE__, __LINE__);
void check1(cudaError_t res, string warningstring, const char *file, int linenum);
//See if the GPU is running correctly
void checkGPUStatus(cudaError_t cudaGetLastError, string warningstring);


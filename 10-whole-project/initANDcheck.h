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
//#include <chrono>//��׼ģ�������ʱ���йص�ͷ�ļ�

using std::string;


//��ӡ�豸��Ϣ
void printDeviceProp(const cudaDeviceProp &prop);
//CUDA ��ʼ��
bool InitCUDA();
void check(cudaError_t res, string warningstring);
//�÷�check1(cudaMalloc((void**)&aa, 10 * sizeof(float)), "aa cudaMalloc Error", __FILE__, __LINE__);
void check1(cudaError_t res, string warningstring, const char *file, int linenum);
//�鿴GPU�����Ƿ���ȷ
void checkGPUStatus(cudaError_t cudaGetLastError, string warningstring);


//#ifndef SIMPLEACQ_UTILITY_KERNELS_H
//#define SIMPLEACQ_UTILITY_KERNELS_H
//
////--- Very simple CUDA kernel to add constant value to input buffer
//__global__ void ImgReverse_kernel(unsigned short * inputBuffer, unsigned short inputInt, int bufferWidth, int bufferHeight); 
//
////--- user function to add constant to input dataset on GPU
////void addConstantOnGpuFunc(unsigned short * inputBuffer, unsigned short inputInt, unsigned short bufferWidth, unsigned short bufferHeight, cudaStream_t *stream); 
//
//#endif
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//--- Very simple CUDA kernel to add constant value to input buffer
__global__ void ImgReverse_kernel(unsigned short* inputBuffer, int bufferWidth, int bufferHeight);
//--- wrapper function for 'addConstant_kernel'
void ImgReverseOnGpuFunc(unsigned short* inputBuffer, unsigned short bufferWidth, unsigned short bufferHeight,
	cudaStream_t* stream);
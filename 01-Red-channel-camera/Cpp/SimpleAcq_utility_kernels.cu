#include "SimpleAcq_utility_kernels.h"
#include <iostream>
#include <cmath>

//--- Very simple CUDA kernel to add constant value to input buffer
__global__ void ImgReverse_kernel(unsigned short * inputBuffer, int bufferWidth, int bufferHeight) 
{
  //--- compute idx, the x and y location of the element in the original array
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;

	unsigned short *pus_ptr1 = inputBuffer;

	if ( idx < bufferWidth && idy < bufferHeight) {
		int i_index = idx + idy*bufferWidth;	
		pus_ptr1[i_index] = (pus_ptr1[i_index] > 255) ? 0 : (255 - pus_ptr1[i_index]);
		/*pus_ptr1[i_index] = pow(pus_ptr1[i_index], 20) / pow(4096, 19);
		pus_ptr1[i_index] = pow(pus_ptr1[i_index], 3) / pow(4096, 2);
		pus_ptr1[i_index] = pow(pus_ptr1[i_index], 2) / pow(4096, 1);*/
		//pus_ptr1[i_index] = ((i_index/16) % 2) ? 100 : 255;
	}
}

//--- wrapper function for 'addConstant_kernel'
void ImgReverseOnGpuFunc(unsigned short * inputBuffer, unsigned short bufferWidth, unsigned short bufferHeight, 
													cudaStream_t *stream) 
{
	//--- Set up CUDA kernel vars
	dim3 threadsPerBlock;
	dim3 blocksPerGrid;

	int devID;
	cudaDeviceProp deviceProp;
	cudaGetDevice(&devID);
	cudaGetDeviceProperties(&deviceProp, devID);

	// Use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;
  
	threadsPerBlock.x = block_size;
	threadsPerBlock.y = block_size;

	blocksPerGrid.x = bufferWidth / threadsPerBlock.x + (bufferWidth % threadsPerBlock.x == 0 ? 0:1);
	blocksPerGrid.y = bufferHeight / threadsPerBlock.y + (bufferHeight % threadsPerBlock.y == 0 ? 0:1);

	//--- call CUDA Kernel
	ImgReverse_kernel<<<blocksPerGrid, threadsPerBlock, 0, *stream>>>(inputBuffer, bufferWidth, bufferHeight);

#ifdef _DEBUG
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();																										
	if( cudaSuccess != err)																																		
	{
		printf("kernel error, err is %s\n", cudaGetErrorString(err));
	}
#endif

};

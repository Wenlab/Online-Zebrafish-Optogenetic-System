#pragma once
//#include "header.cuh"
//#include "initANDcheck.h"
#include "reconstructionCUDA.cuh"



__global__ void Zhuan_Complex_kernel(float *PSF_1_gpu, cufftComplex *PSF_1_gpu_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		PSF_1_gpu_Complex[i].x = PSF_1_gpu[i];
		PSF_1_gpu_Complex[i].y = 0;
	}
}
__global__ void PSF_unshort(float *PSF_1_gpu, unsigned short *PSF, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if (PSF_1_gpu[i] < 0)
		{
			PSF[i] = 0;
		}
		else if (PSF_1_gpu[i] > 65535)
		{
			PSF[i] = 65535;
		}
		else
		{
			PSF[i] = (int)(PSF_1_gpu[i] + 0.5);
		}
	}
}
__global__ void initial_kernel_1(float *ImgEst, float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		ImgEst[i] = 0;
		Ratio[i] = 1;
	}
}
__global__ void gpuObjRecon_fuzhi(float *gpuObjRecon, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecon[i] = 1;
	}
}
__global__ void initial_kernel_3(float *gpuObjRecROI, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecROI[i] = 1;
	}
}
__global__ void ImgExp_ge(unsigned short *Img_gpu, int BkgMean, float *ImgExp, int total)
{
	//Turn the result of the difference less than 0 into 0, greater than 0 rounded, greater than 65535 into 65535
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if ((Img_gpu[i] - BkgMean) < 0)
		{
			ImgExp[i] = 0;
		}
		else if ((Img_gpu[i] - BkgMean) > 65535)
		{
			ImgExp[i] = 65535;
		}
		else
		{
			ImgExp[i] = (int)((Img_gpu[i] - BkgMean) + 0.5);
		}
	}
}
__global__ void Ratio_fuzhi(float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio[i] = 1;
	}
}
__global__ void OTF_mul_gpuObjRecon_Complex(cufftComplex *OTF, cufftComplex *gpuObjRecon_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		float aaa = OTF[i].x*gpuObjRecon_Complex[i].x - OTF[i].y*gpuObjRecon_Complex[i].y;//Real Department Results
		float bbb = OTF[i].x*gpuObjRecon_Complex[i].y + OTF[i].y*gpuObjRecon_Complex[i].x;//Virtual Part Results
		gpuObjRecon_Complex[i].x = aaa;
		gpuObjRecon_Complex[i].y = bbb;
	}
}
__global__ void ifftshift_real_max(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	int lie_half = PSF_size_2 / 2;
	if (i < PSF_size_1 / 2 && j < PSF_size_2 && k < PSF_size_3)
	{
		//Implement the image ifftshift+real+max, i.e.: divide the image into 4 quadrants, swap the first and third translations, swap the second and fourth translations
		float_temp[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * 512] = OTF[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j].x >= 0 ? OTF[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j].x : 0;
		float_temp[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j] = OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2].x >= 0 ? OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2].x : 0;
	}
}
__global__ void ifftshift(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3, cufftComplex *OTF_ifftshift)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	int lie_half = PSF_size_2 / 2;
	if (i < PSF_size_1 / 2 && j < PSF_size_2 && k < PSF_size_3)
	{
		//Implement the image ifftshift, i.e.: divide the image into 4 quadrants, first and third translation swap, second and fourth translation swap
		OTF_ifftshift[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * 512] = OTF[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j];
		OTF_ifftshift[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j] = OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2];
	}
}
__global__ void float_temp_sum(float *float_temp, float *ImgEst, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < PSF_size_1 && j < PSF_size_2)
	{
		ImgEst[i*PSF_size_2 + j] = 0;
		for (int k = 0; k < PSF_size_3; k++)
		{
			ImgEst[i*PSF_size_2 + j] += float_temp[k*PSF_size_1*PSF_size_2 + (i*PSF_size_2 + j)];
		}
	}
}
__global__ void Ratio_fuzhi_2(float *ImgExp, float *ImgEst, float Tmp, int SNR, float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio[i] = ImgExp[i] / (ImgEst[i] + Tmp / SNR);
	}
}
__global__ void Ratio_Complex_ge(float *ImgExp, float *ImgEst, float Tmp, int SNR, cufftComplex *Ratio_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio_Complex[i].x = ImgExp[i] / (ImgEst[i] + Tmp / SNR);
		Ratio_Complex[i].y = 0;
	}
}
__global__ void fftRatio_ge(cufftComplex *Ratio_Complex, cufftComplex *fftRatio, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < PSF_size_1 && j < PSF_size_2 && k < PSF_size_3)
	{
		fftRatio[k*PSF_size_1*PSF_size_2 + i * PSF_size_2 + j] = Ratio_Complex[i*PSF_size_2 + j];
	}
}
__global__ void fftceshi_gpu_fuzhi(cufftComplex *PSF_1_gpu_Complex, cufftComplex *fftceshi_gpu, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		fftceshi_gpu[i] = PSF_1_gpu_Complex[i];
	}
}
__global__ void ifft2_divide(cufftComplex *OTF, int total, int scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		OTF[i].x = OTF[i].x / scale;
		OTF[i].y = OTF[i].y / scale;
	}
}
__global__ void real_multiply(float *gpuObjRecon, float *float_temp, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecon[i] = gpuObjRecon[i] * float_temp[i];
	}
}
__global__ void fftRatio_mul_conjOTF(cufftComplex *fftRatio, cufftComplex *OTF, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		float aaa = fftRatio[i].x*OTF[i].x + fftRatio[i].y*OTF[i].y;//Real Department Results
		float bbb = -fftRatio[i].x*OTF[i].y + fftRatio[i].y*OTF[i].x;//Virtual Part Results
		fftRatio[i].x = aaa;
		fftRatio[i].y = bbb;
	}
}


__global__ void cropReconImage_kernel(float *gpuObjRecon, float *gpuObjRecon_crop)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;//XObj
	const int y = blockDim.y * blockIdx.y + threadIdx.y;//YObj
	const int z = blockDim.z * blockIdx.z + threadIdx.z;//ZObj

	int line_start = 156;
	int line_end = 355;
	int line_total = 200;
	int col_start = 156;
	int col_end = 355;
	int col_total = 200;
	int band = 50;

	if (z < 50 && x < 200 && y < 200)
	{
		gpuObjRecon_crop[z * 200 * 200 + y * 200 + x] = gpuObjRecon[z*512*512 + (y + line_start)*255 + x + col_start];
		//gpuObjRecon_crop[z*200*200 + y * 200 + x] = gpuObjRecon[z * 512 * 512 + (256 - 100 + y) * 512 + 256 - 100 + x];
	}
}
#include"templateMatchingCUDA.cuh"
#include"initANDcheck.h"



using namespace std;
using namespace chrono;
//图像几何变换C++实现--镜像，平移，旋转，错切，缩放
//https://blog.csdn.net/duiwangxiaomi/article/details/109532590


__global__ void kernel_1(float *ObjRecon_gpu, int height, int width, float *image2D_XY_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//行循环
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//列循环

	if (i < 200 && j < 200)
	{
		image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[i * 200 + j];
		for (int b = 0; b < 50; b++)//波段循环
		{
			if (image2D_XY_gpu[i * 200 + j] < ObjRecon_gpu[b * 200 * 200 + i * 200 + j])
			{
				image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[b * 200 * 200 + i * 200 + j];
			}
		}//波段循环
	}
}
__global__ void kernel_2(float *image2D_XY_gpu, int total, double image2D_XY_mean, float *img2DBW_XY_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if (image2D_XY_gpu[i] > image2D_XY_mean)
			img2DBW_XY_gpu[i] = 1.0;
			//img2DBW_XY_gpu[i] = 255;
		else
			img2DBW_XY_gpu[i] = 0.0;
	}

}
__global__ void kernel_3(float *template_roXY_gpu, float *img2DBW_XY_gpu, int rotationAngleXY_size, double *err_XY_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < rotationAngleXY_size)
	{
		//计算两个矩阵的均方误差
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//行循环
		{
			for (int k = 0; k < 200; k++)//列循环
			{
				sum_temp += (template_roXY_gpu[i * 200 * 200 + j * 200 + k] - img2DBW_XY_gpu[j * 200 + k])*
					(template_roXY_gpu[i * 200 * 200 + j * 200 + k] - img2DBW_XY_gpu[j * 200 + k]);
			}
		}
		err_XY_gpu[i] = sum_temp / (200 * 200);
	}
}

__global__ void kernel_4(float *imageRotated3D_gpu, float *image2D_YZ_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//波段循环
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//行循环

	if (i < 50 && j < 200)
	{
		image2D_YZ_gpu[i * 200 + j] = -FLT_MAX;
		for (int k = 0; k < 200; k++)//列循环，求一行的最大值
		{
			if (image2D_YZ_gpu[i * 200 + j] < imageRotated3D_gpu[i * 200 * 200 + j * 200 + k])
			{
				image2D_YZ_gpu[i * 200 + j] = imageRotated3D_gpu[i * 200 * 200 + j * 200 + k];
			}
		}
	}
}
__global__ void kernel_5(float *image2D_YZ_gpu, double image2D_YZ_mean, float *img2DBW_YZ_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < 200 * 50)
	{
		if (image2D_YZ_gpu[i] > image2D_YZ_mean)
			img2DBW_YZ_gpu[i] = 1.0;
		else
			img2DBW_YZ_gpu[i] = 0.0;
	}
}
__global__ void kernel_6(float *template_roYZ_gpu, float *img2DBW_YZ_gpu, int rotationAngleYZ_size, double *err_YZ_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < rotationAngleYZ_size)
	{
		//计算两个矩阵的均方误差
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//行循环
		{
			for (int k = 0; k < 50; k++)//列循环
			{
				//template_roYZ是200行*50列*31波段，行优先排列，img2DBW_YZ是列优先排列的
				sum_temp += (template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j])*
					(template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j]);
			}
		}
		err_YZ_gpu[i] = sum_temp / (200 * 50);
	}
}
//维度变换
__global__ void kernel_7(float *imageRotated3D_gpu, float *imageRotated3D_gpu_1)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//输出波段循环，输入的列循环
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//输出行循环，输入的行循环，反着来
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//输出列循环，输入的波段循环

	if (i < 200 && j < 200 && k < 50)
	{
		//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
		imageRotated3D_gpu_1[i * 200 * 50 + j * 50 + k] = imageRotated3D_gpu[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
	}
}
//按照X轴旋转
//void ObjRecon_imrotate3_X_gpu(float *imageRotated3D_gpu_1, double nAngle, float *imageRotated3D_gpu_2)
//{
//	NppiSize Input_Size;//输入图像的行列数
//	Input_Size.width = 200;
//	Input_Size.height = 50;
//	/* 分配显存，将原图传入显存 */
//	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//每行所占的字节数
//	float *input_image_gpu;
//	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);
//
//
//	/* 计算旋转后长宽 */
//	NppiRect Input_ROI;//特定区域的旋转，相当于裁剪图像的一块，本次采用全部图像
//	Input_ROI.x = Input_ROI.y = 0;
//	Input_ROI.width = Input_Size.width;
//	Input_ROI.height = Input_Size.height;
//	double aBoundingBox[2][2];
//	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
//	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//起始列
//	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//起始行
//	aBoundingBox[0][0] = bb;//起始列
//	aBoundingBox[0][1] = cc;//起始行
//	NppiSize Output_Size;
//	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
//	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
//	Output_Size.width = Input_Size.width;
//	Output_Size.height = Input_Size.height;
//
//
//	/* 转换后的图像显存分配 */
//	int nDstPitchCUDA = Output_Size.width * sizeof(float);
//	float *output_image_gpu;
//	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);
//
//
//	//输出感兴趣区的大小，相当于把输出图像再裁剪一遍，应该是这样，还没测试，这个有用
//	NppiRect Output_ROI;
//	Output_ROI.x = 0; Output_ROI.y = 0;
//	Output_ROI.width = Input_Size.width;
//	Output_ROI.height = Input_Size.height;
//
//	for (int i = 0; i < 200; i++)
//	{
//		check(cudaMemcpy(input_image_gpu, imageRotated3D_gpu_1 + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
//		/* 处理旋转 */
//		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
//			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
//		assert(nppRet == NPP_NO_ERROR);
//		check(cudaMemcpy(imageRotated3D_gpu_2 + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
//	}
//}
//再变换到原来的维度
__global__ void kernel_8(float *imageRotated3D_gpu_2, float *imageRotated3D_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//输出波段循环，输入的列循环
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//输出行循环，输入的行循环，反着来
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//输出列循环，输入的波段循环

	if (i < 200 && j < 200 && k < 50)//输出波段循环，输入的列循环
	{
		imageRotated3D_gpu[(49 - k) * 200 * 200 + (199 - j) * 200 + i] = imageRotated3D_gpu_2[i * 200 * 50 + j * 50 + k];
	}
}
__global__ void kernel_9(float *imageRotated3D_gpu, double imageRotated3D_x_mean, int *BWObjRecon_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < 200 * 200 * 50)
	{
		if (imageRotated3D_gpu[i] > imageRotated3D_x_mean)
			BWObjRecon_gpu[i] = 1;
		else
			BWObjRecon_gpu[i] = 0;
	}
}
__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;//XObj
	const int y = blockDim.y * blockIdx.y + threadIdx.y;//YObj
	const int z = blockDim.z * blockIdx.z + threadIdx.z;//ZObj

	if (z < ZObj && x < XObj && y < YObj)
	{
		ObjReconRed_gpu[z*XObj*YObj + y * XObj + x] = imageRotated3D_gpu[z * 200 * 200 + (CentroID0 - 61 + y) * 200 + CentroID2 - 38 + x];
	}

}
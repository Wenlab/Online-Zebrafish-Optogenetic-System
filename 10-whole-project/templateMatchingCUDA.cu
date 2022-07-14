#include"templateMatchingCUDA.cuh"
#include"initANDcheck.h"



using namespace std;
using namespace chrono;
//ͼ�񼸺α任C++ʵ��--����ƽ�ƣ���ת�����У�����
//https://blog.csdn.net/duiwangxiaomi/article/details/109532590


__global__ void kernel_1(float *ObjRecon_gpu, int height, int width, float *image2D_XY_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//��ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//��ѭ��

	if (i < 200 && j < 200)
	{
		image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[i * 200 + j];
		for (int b = 0; b < 50; b++)//����ѭ��
		{
			if (image2D_XY_gpu[i * 200 + j] < ObjRecon_gpu[b * 200 * 200 + i * 200 + j])
			{
				image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[b * 200 * 200 + i * 200 + j];
			}
		}//����ѭ��
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
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 200; k++)//��ѭ��
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
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//����ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//��ѭ��

	if (i < 50 && j < 200)
	{
		image2D_YZ_gpu[i * 200 + j] = -FLT_MAX;
		for (int k = 0; k < 200; k++)//��ѭ������һ�е����ֵ
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
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 50; k++)//��ѭ��
			{
				//template_roYZ��200��*50��*31���Σ����������У�img2DBW_YZ�����������е�
				sum_temp += (template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j])*
					(template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j]);
			}
		}
		err_YZ_gpu[i] = sum_temp / (200 * 50);
	}
}
//ά�ȱ任
__global__ void kernel_7(float *imageRotated3D_gpu, float *imageRotated3D_gpu_1)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//�������ѭ�����������ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//�����ѭ�����������ѭ����������
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//�����ѭ��������Ĳ���ѭ��

	if (i < 200 && j < 200 && k < 50)
	{
		//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
		imageRotated3D_gpu_1[i * 200 * 50 + j * 50 + k] = imageRotated3D_gpu[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
	}
}
__global__ void kernel_8(float *imageRotated3D_gpu_2, float *imageRotated3D_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//�������ѭ�����������ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//�����ѭ�����������ѭ����������
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//�����ѭ��������Ĳ���ѭ��

	if (i < 200 && j < 200 && k < 50)//�������ѭ�����������ѭ��
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
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <tchar.h>
#include <io.h>
#include <string>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <FreeImage.h>
#include <npp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h> 
#include <thrust/sort.h> 
#include <thrust/copy.h> 
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <chrono>//标准模板库中与时间有关的头文件
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>
#include "gdal_mdreader.h"
#include "gdalwarper.h"
#include "ogrsf_frmts.h"

using namespace std;
using namespace chrono;
//图像几何变换C++实现--镜像，平移，旋转，错切，缩放
//https://blog.csdn.net/duiwangxiaomi/article/details/109532590


int clockRate = 1.0;
//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("GPU Parament：\n");
	printf(" Device Name : %s.\n", prop.name);
	printf(" totalGlobalMem : %I64d.\n", prop.totalGlobalMem);
	printf(" sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf(" regsPerBlock : %d.\n", prop.regsPerBlock);
	printf(" warpSize : %d.\n", prop.warpSize);
	printf(" memPitch : %d.\n", prop.memPitch);
	printf(" maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf(" maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf(" maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf(" totalConstMem : %d.\n", prop.totalConstMem);
	printf(" major.minor : %d.%d.\n", prop.major, prop.minor);
	printf(" clockRate : %d.\n", prop.clockRate);
	printf(" textureAlignment : %d.\n", prop.textureAlignment);
	printf(" deviceOverlap : %d.\n", prop.deviceOverlap);
	printf(" multiProcessorCount : %d.\n", prop.multiProcessorCount);
	std::printf(" CUDA core: %d\r\n", 2 * prop.multiProcessorCount* prop.maxThreadsPerMultiProcessor / prop.warpSize);
	printf("\n\n");
}
//CUDA 初始化
bool InitCUDA()
{
	int count;
	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//打印设备信息
		printDeviceProp(prop);
		//获得显卡的时钟频率
		clockRate = prop.clockRate;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}
void check(cudaError_t res, string warningstring)
{
	if (res != cudaSuccess)
	{
		printf((warningstring + " !\n").c_str());
		system("pause");
		exit(0);
	}
}
void check1(cudaError_t res, string warningstring, const char *file, int linenum)
{
	//用法check1(cudaMalloc((void**)&aa, 10 * sizeof(float)), "aa cudaMalloc Error", __FILE__, __LINE__);
	if (res != cudaSuccess)
	{
		printf((warningstring + " !\n").c_str());
		printf("   Error text: %s   Error code: %d\n", cudaGetErrorString(res), res);
		printf("   Line:    %d    File:    %s\n", linenum, file);
		system("pause");
		exit(0);
	}
}
//查看GPU运行是否正确
void checkGPUStatus(cudaError_t cudaGetLastError, string warningstring)
{
	if (cudaGetLastError != cudaSuccess)
	{
		printf("\n\n");
		printf((warningstring + " !\n").c_str());
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError));
		system("pause");
		exit(0);
	}
}

//npp的图像旋转函数nppiRotate_32f_C1R使用方法：
//若旋转角度为正，图像从左下角行优先开始读的：顺时针旋转。左上角开始读的：逆时针旋转
//若旋转角度为负，图像从左下角行优先开始读的：逆时针旋转。左上角开始读的：顺时针旋转
float *ObjRecon_imrotate3(float *ObjRecon, double nAngle)
{
	//float *input_image = new float[200*200];
	float *imageRotated3D = new float[200 * 200 * 50];
	//for (int i = 0; i < 200 * 200; i++)
	//{
	//	input_image[i] = ObjRecon[i];
	//}

	NppiSize Input_Size;//输入图像的行列数
	Input_Size.width = 200;
	Input_Size.height = 200;
	/* 分配显存，将原图传入显存 */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//每行所占的字节数
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);
	//check(cudaMemcpy(input_image_gpu, input_image, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");


	/* 计算旋转后长宽 */
	NppiRect Input_ROI;//特定区域的旋转，相当于裁剪图像的一块，本次采用全部图像
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//起始列
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//起始行
	aBoundingBox[0][0] = bb;//起始列
	aBoundingBox[0][1] = cc;//起始行
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* 转换后的图像显存分配 */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//输出感兴趣区的大小，相当于把输出图像再裁剪一遍，应该是这样，还没测试，这个有用
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 50; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* 处理旋转 */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	////旋转前后的第一个波段分别写出来看看
	//float *ObjRecon_1 = new float[200 * 200];
	//float *imageRotated3D_1 = new float[200 * 200];
	//for (int i = 0; i < 200*200; i++)
	//{
	//	ObjRecon_1[i] = ObjRecon[i];
	//	imageRotated3D_1[i] = imageRotated3D[i];
	//}
	//GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	//GDALDataset *ds1 = pDriver->Create("ObjRecon_1_c", 200, 200, 1, GDT_Float32, NULL);
	//GDALDataset *ds2 = pDriver->Create("imageRotated3D_1_c", 200, 200, 1, GDT_Float32, NULL);
	//if ((ds1 == NULL) || (ds2 == NULL))
	//{
	//	cout << "create ObjRecon_1 imageRotated3D_1 output_file error!" << endl;
	//	system("pause");
	//	return 0;
	//}
	////从图像的左上角一行一行的写
	//ds1->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, ObjRecon_1, 200, 200, GDT_Float32, 0, 0);
	//ds2->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, imageRotated3D_1, 200, 200, GDT_Float32, 0, 0);
	//GDALClose(ds1);
	//GDALClose(ds2);



	return imageRotated3D;
}

//按照X轴旋转
float *ObjRecon_imrotate3_X(float *imageRotated3D, double nAngle)
{
	//imageRotated3D(200*200*50)转换成：列变成波段，波段变成列，行变成反着，变成200行*50列*200波段
	float *ObjRecon = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//输出波段循环，输入的列循环
	{
		for (int j = 0; j < 200; j++)//输出行循环，输入的行循环，反着来
		{
			for (int k = 0; k < 50; k++)//输出列循环，输入的波段循环
			{
				//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
				ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
			}
		}
	}

	float *imageRotated3D_rotate = new float[200 * 50 * 200];

	NppiSize Input_Size;//输入图像的行列数
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* 分配显存，将原图传入显存 */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//每行所占的字节数
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* 计算旋转后长宽 */
	NppiRect Input_ROI;//特定区域的旋转，相当于裁剪图像的一块，本次采用全部图像
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//起始列
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//起始行
	aBoundingBox[0][0] = bb;//起始列
	aBoundingBox[0][1] = cc;//起始行
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* 转换后的图像显存分配 */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//输出感兴趣区的大小，相当于把输出图像再裁剪一遍，应该是这样，还没测试，这个有用
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* 处理旋转 */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_rotate + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	////旋转前后的第一个波段分别写出来看看
	//float *ObjRecon_1 = new float[200 * 200];
	//float *imageRotated3D_1 = new float[200 * 200];
	//for (int i = 0; i < 200*200; i++)
	//{
	//	ObjRecon_1[i] = ObjRecon[i];
	//	imageRotated3D_1[i] = imageRotated3D[i];
	//}
	//GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	//GDALDataset *ds1 = pDriver->Create("ObjRecon_1_c", 200, 200, 1, GDT_Float32, NULL);
	//GDALDataset *ds2 = pDriver->Create("imageRotated3D_1_c", 200, 200, 1, GDT_Float32, NULL);
	//if ((ds1 == NULL) || (ds2 == NULL))
	//{
	//	cout << "create ObjRecon_1 imageRotated3D_1 output_file error!" << endl;
	//	system("pause");
	//	return 0;
	//}
	////从图像的左上角一行一行的写
	//ds1->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, ObjRecon_1, 200, 200, GDT_Float32, 0, 0);
	//ds2->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, imageRotated3D_1, 200, 200, GDT_Float32, 0, 0);
	//GDALClose(ds1);
	//GDALClose(ds2);

	//再变换到原来的维度分布
	//200行*50列*200波段—>>200行*200行*50列
	float *imageRotated3D_rotate_return = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//输出波段循环，输入的列循环
	{
		for (int j = 0; j < 200; j++)//输出行循环，输入的行循环，反着来
		{
			for (int k = 0; k < 50; k++)//输出列循环，输入的波段循环
			{
				imageRotated3D_rotate_return[(49 - k) * 200 * 200 + (199 - j) * 200 + i] = imageRotated3D_rotate[i * 200 * 50 + j * 50 + k];
			}
		}
	}

	return imageRotated3D_rotate_return;
}

//cpu版本
int main0()
{
	//开始计时
	auto time_start = system_clock::now();
	GDALAllRegister();
	//设置支持中文路径
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	OGRRegisterAll();
	//CUDA 初始化
	if (!InitCUDA())
	{
		cout << "CUDA支持下的显卡设备初始化失败!" << endl;
		system("pause");
		return 0;
	}

	const char *rotationAngleXY_file = "F:/Archive/rotationAngleXY.dat";//360个double
	const char *rotationAngleYZ_file = "F:/Archive/rotationAngleYZ.dat";//31个double
	const char *template_roXY_file = "F:/Archive/template_roXY.dat";//200*200*360个float，按照matlab中行优先存储，存完一个波段再存第二个波段
	const char *template_roYZ_file = "F:/Archive/template_roYZ.dat";//200*50*31个float，按照matlab中行优先存储，存完一个波段再存第二个波段
	const char *ObjRecon_file = "F:/Archive/ObjRecon.dat";//200*200*50个float，按照matlab中行优先存储，存完一个波段再存第二个波段

	FILE * rotationAngleXY_fid = fopen(rotationAngleXY_file, "rb");
	if (rotationAngleXY_fid == NULL)
	{
		cout << rotationAngleXY_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int rotationAngleXY_size = 360;
	double *rotationAngleXY = new double[rotationAngleXY_size];
	fread(rotationAngleXY, sizeof(double), rotationAngleXY_size, rotationAngleXY_fid);
	fclose(rotationAngleXY_fid);
	FILE * rotationAngleYZ_fid = fopen(rotationAngleYZ_file, "rb");
	if (rotationAngleYZ_fid == NULL)
	{
		cout << rotationAngleYZ_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int rotationAngleYZ_size = 31;
	double *rotationAngleYZ = new double[rotationAngleYZ_size];
	fread(rotationAngleYZ, sizeof(double), rotationAngleYZ_size, rotationAngleYZ_fid);
	fclose(rotationAngleYZ_fid);
	FILE * template_roXY_fid = fopen(template_roXY_file, "rb");
	if (template_roXY_fid == NULL)
	{
		cout << template_roXY_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int template_roXY_size = 200*200*360;
	float *template_roXY = new float[template_roXY_size];
	fread(template_roXY, sizeof(float), template_roXY_size, template_roXY_fid);
	fclose(template_roXY_fid);
	FILE * template_roYZ_fid = fopen(template_roYZ_file, "rb");
	if (template_roYZ_fid == NULL)
	{
		cout << template_roYZ_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int template_roYZ_size = 200 * 50 * 31;
	float *template_roYZ = new float[template_roYZ_size];
	fread(template_roYZ, sizeof(float), template_roYZ_size, template_roYZ_fid);
	fclose(template_roYZ_fid);
	FILE * ObjRecon_fid = fopen(ObjRecon_file, "rb");
	if (ObjRecon_fid == NULL)
	{
		cout << ObjRecon_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int ObjRecon_size = 200 * 200 * 50;
	float *ObjRecon = new float[ObjRecon_size];
	fread(ObjRecon, sizeof(float), ObjRecon_size, ObjRecon_fid);
	fclose(ObjRecon_fid);

	//计算ObjRecon一个像素在所有波段中的最大值，按照matlab中的矩阵行优先存储
	float *image2D_XY = new float[200 * 200];//按照行优先排列
	double image2D_XY_sum = 0;
	for (int i = 0; i < 200; i++)//行循环
	{
		for (int j = 0; j < 200; j++)//列循环
		{
			image2D_XY[i * 200 + j] = ObjRecon[i * 200 + j];
			for (int b = 0; b < 50; b++)//波段循环
			{
				if (image2D_XY[i * 200 + j] < ObjRecon[b * 200 * 200 + i * 200 + j])
				{
					image2D_XY[i * 200 + j] = ObjRecon[b * 200 * 200 + i * 200 + j];
				}
			}//波段循环
			image2D_XY_sum += image2D_XY[i * 200 + j];
		}
	}
	// 对投影二值化: 大于mean的取1， 小于等于mean的取0
	double image2D_XY_mean = image2D_XY_sum / (200 * 200);
	float *img2DBW_XY = new float[200 * 200]();
	for (int i = 0; i < 200*200; i++)
	{
		if (image2D_XY[i] > image2D_XY_mean)
			img2DBW_XY[i] = 1.0;
		else
			img2DBW_XY[i] = 0.0;
	}

	// 对每个角度的误差 初始化
	double *err_XY = new double[rotationAngleXY_size];
	double err_XY_min = DBL_MAX;
	//求二值化结果对于每一个角度的误差,GPU中可以直接并发一次执行这个循环
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		//计算两个矩阵的均方误差
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//行循环
		{
			for (int k = 0; k < 200; k++)//列循环
			{
				sum_temp += (template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k])*(template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k]);
			}
		}
		err_XY[i] = sum_temp / (200 * 200);
		if (err_XY[i] < err_XY_min)
			err_XY_min = err_XY[i];
	}
	//找到最小值对应的索引
	int idx;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] == err_XY_min)
		{
			idx = i;
			break;
		}
	}
	
	//ObjRecon（200*200*50个float，行优先排列），绕Z轴顺时针旋转rotationAngleXY[idx]度
	//旋转角度为正：逆时针，负：顺时针
	float *imageRotated3D = ObjRecon_imrotate3(ObjRecon, -rotationAngleXY[idx]);

	/* Y - Z rotation */
	//求 y-z面的投影,计算imageRotated3D一个像素在行方向的最大值
	float *image2D_YZ = new float[200 * 50];//200行*50列按照imageRotated3D的列优先排列，是matlab中按照列优先排列
	double image2D_YZ_sum = 0;
	for (int i = 0; i < 50; i++)//波段循环
	{
		for (int j = 0; j < 200; j++)//行循环
		{
			image2D_YZ[i * 200 + j] = -FLT_MAX;
			for (int k = 0; k < 200; k++)//列循环，求一行的最大值
			{
				if (image2D_YZ[i * 200 + j] < imageRotated3D[i * 200 * 200 + j * 200 + k])
				{
					image2D_YZ[i * 200 + j] = imageRotated3D[i * 200 * 200 + j * 200 + k];
				}
			}
			image2D_YZ_sum += image2D_YZ[i * 200 + j];
		}
	}
	double image2D_YZ_mean = image2D_YZ_sum / (200 * 50) + 14;

	//二值化 y-z面，大于mean的取1， 小于等于mean的取0
	float *img2DBW_YZ = new float[200 * 50];
	for (int i = 0; i < 200 * 50; i++)
	{
		if (image2D_YZ[i] > image2D_YZ_mean)
			img2DBW_YZ[i] = 1.0;
		else
			img2DBW_YZ[i] = 0.0;
	}

	//对每个角度的误差 初始化
	double *err_YZ = new double[rotationAngleYZ_size];
	double err_YZ_min = DBL_MAX;
	//求二值化结果对于每一个角度的误差，GPU中可以直接并发一次执行这个循环
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		//计算两个矩阵的均方误差
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//行循环
		{
			for (int k = 0; k < 50; k++)//列循环
			{
				//template_roYZ是200行*50列*31波段，行优先排列，img2DBW_YZ是列优先排列的
				sum_temp += (template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j])*(template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j]);
			}
		}
		err_YZ[i] = sum_temp / (200 * 50);
		if (err_YZ[i] < err_YZ_min)
			err_YZ_min = err_YZ[i];
	}
	//找到最小值对应的索引
	int idx2;
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		if (err_YZ[i] == err_YZ_min)
		{
			idx2 = i;
			break;
		}
	}

	//imageRotated3D旋转，绕X轴逆旋转rotationAngleYZ(idx2)度
	float *imageRotated3D_x = ObjRecon_imrotate3_X(imageRotated3D, rotationAngleYZ[idx2]);

	// Crop Out，把旋转完的3D图像中的鱼切出来
	// 二值化旋转后的图像
	//计算imageRotated3D_x的均值
	double imageRotated3D_x_sum = 0;
	for (int i = 0; i < 200*200*50; i++)
	{
		imageRotated3D_x_sum += imageRotated3D_x[i];
	}
	double imageRotated3D_x_mean = imageRotated3D_x_sum / (200 * 200 * 50) + 4;

	int *BWObjRecon = new int[200 * 200 * 50]();
	int *idx_2 = new int[200 * 200 * 50]();//imageRotated3D_x大于均值的索引
	int idx_2_size = 0;
	for (int i = 0; i < 200 * 200 * 50; i++)
	{
		if (imageRotated3D_x[i] > imageRotated3D_x_mean)
		{
			idx_2_size++;
			idx_2[idx_2_size] = i;
			BWObjRecon[i] = 1;
		}
		else
			BWObjRecon[i] = 0;
	}
	//idx_2数组的每一个数字转换成BWObjRecon（200行*200列*50波段）的行号，列号，波段号
	float *x = new float[idx_2_size]; float x_sum = 0;
	float *y = new float[idx_2_size]; float y_sum = 0;
	float *z = new float[idx_2_size]; float z_sum = 0;
	for (int i = 0; i < idx_2_size; i++)
	{
		z[i] = idx_2[i] / (200 * 200);
		int yushu = idx_2[i] % (200 * 200);
		x[i] = yushu / 200;
		y[i] = yushu % 200;

		x_sum += x[i];
		y_sum += y[i];
		z_sum += z[i];
	}
	int CentroID[3];
	CentroID[0] = int(x_sum / idx_2_size + 0.5);
	CentroID[1] = int(y_sum / idx_2_size + 0.5);
	CentroID[2] = int(z_sum / idx_2_size + 0.5);
	//CentroID数组在matlab中是[89,91,24]，我计算的是[86,91,24],x相差3，是npp旋转和matlab的结果有误差造成的，误差也不大，算正常！！

	// 保留质心坐标周围的区域，坐标索引在matlab的数上要减去1
	// 行范围：【CentroID(0)-61：CentroID(0)+33】 。列范围：【CentroID(2)-38：CentroID(2)+37】。所有的波段
	int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//行
	int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//列
	int	ZObj = 50;//波段
	float *ObjReconRed = new float[XObj*YObj*ZObj];
	for (int i = 0; i < ZObj; i++)//波段循环
	{
		for (int j = 0; j < XObj; j++)//行循环
		{
			for (int k = 0; k < YObj; k++)//列循环
			{
				ObjReconRed[i*XObj*YObj + j*YObj + k] =
					imageRotated3D_x[i*200*200 + (CentroID[0] - 61 + j)*200 + CentroID[2] - 38 + k];
			}
		}
	}
	// size of reference atlas
	//int XRef = 95; int YRef = 76; int ZRef = 50;

	//不再做matlab中的interp3处理，所以数组ObjReconRed就是最后的结果，即matlab的RescaledRed数组



	auto time_end = system_clock::now();
	auto duration = duration_cast<microseconds>(time_end - time_start);
	float usetime_total = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	cout << "finish，用时：" << usetime_total << endl;
	system("pause");
    return 0;
}


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
void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu)
{
	NppiSize Input_Size;//输入图像的行列数
	Input_Size.width = 200;
	Input_Size.height = 200;
	/* 分配显存，将原图传入显存 */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//每行所占的字节数
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* 计算旋转后长宽 */
	NppiRect Input_ROI;//特定区域的旋转，相当于裁剪图像的一块，本次采用全部图像
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//起始列
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//起始行
	aBoundingBox[0][0] = bb;//起始列
	aBoundingBox[0][1] = cc;//起始行
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* 转换后的图像显存分配 */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//输出感兴趣区的大小，相当于把输出图像再裁剪一遍，应该是这样，还没测试，这个有用
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 50; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon_gpu + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
		/* 处理旋转 */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_gpu + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
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
void ObjRecon_imrotate3_X_gpu(float *imageRotated3D_gpu_1, double nAngle, float *imageRotated3D_gpu_2)
{
	NppiSize Input_Size;//输入图像的行列数
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* 分配显存，将原图传入显存 */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//每行所占的字节数
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* 计算旋转后长宽 */
	NppiRect Input_ROI;//特定区域的旋转，相当于裁剪图像的一块，本次采用全部图像
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//起始列
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//起始行
	aBoundingBox[0][0] = bb;//起始列
	aBoundingBox[0][1] = cc;//起始行
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* 转换后的图像显存分配 */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//输出感兴趣区的大小，相当于把输出图像再裁剪一遍，应该是这样，还没测试，这个有用
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, imageRotated3D_gpu_1 + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
		/* 处理旋转 */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_gpu_2 + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
	}
}
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


//gpu版本
int main()
{
	//开始计时
	auto time_start = system_clock::now();
	GDALAllRegister();
	//设置支持中文路径
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	OGRRegisterAll();
	//CUDA 初始化
	if (!InitCUDA())
	{
		cout << "CUDA支持下的显卡设备初始化失败!" << endl;
		system("pause");
		return 0;
	}

	const char *rotationAngleXY_file = "F:/Archive/rotationAngleXY.dat";//360个double
	const char *rotationAngleYZ_file = "F:/Archive/rotationAngleYZ.dat";//31个double
	const char *template_roXY_file = "F:/Archive/template_roXY.dat";//200*200*360个float，按照matlab中行优先存储，存完一个波段再存第二个波段
	const char *template_roYZ_file = "F:/Archive/template_roYZ.dat";//200*50*31个float，按照matlab中行优先存储，存完一个波段再存第二个波段
	const char *ObjRecon_file = "F:/Archive/ObjRecon.dat";//200*200*50个float，按照matlab中行优先存储，存完一个波段再存第二个波段

	FILE * rotationAngleXY_fid = fopen(rotationAngleXY_file, "rb");
	if (rotationAngleXY_fid == NULL)
	{
		cout << rotationAngleXY_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int rotationAngleXY_size = 360;
	double *rotationAngleXY = new double[rotationAngleXY_size];
	fread(rotationAngleXY, sizeof(double), rotationAngleXY_size, rotationAngleXY_fid);
	fclose(rotationAngleXY_fid);
	FILE * rotationAngleYZ_fid = fopen(rotationAngleYZ_file, "rb");
	if (rotationAngleYZ_fid == NULL)
	{
		cout << rotationAngleYZ_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int rotationAngleYZ_size = 31;
	double *rotationAngleYZ = new double[rotationAngleYZ_size];
	fread(rotationAngleYZ, sizeof(double), rotationAngleYZ_size, rotationAngleYZ_fid);
	fclose(rotationAngleYZ_fid);
	FILE * template_roXY_fid = fopen(template_roXY_file, "rb");
	if (template_roXY_fid == NULL)
	{
		cout << template_roXY_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int template_roXY_size = 200 * 200 * 360;
	float *template_roXY = new float[template_roXY_size];
	fread(template_roXY, sizeof(float), template_roXY_size, template_roXY_fid);
	fclose(template_roXY_fid);
	FILE * template_roYZ_fid = fopen(template_roYZ_file, "rb");
	if (template_roYZ_fid == NULL)
	{
		cout << template_roYZ_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int template_roYZ_size = 200 * 50 * 31;
	float *template_roYZ = new float[template_roYZ_size];
	fread(template_roYZ, sizeof(float), template_roYZ_size, template_roYZ_fid);
	fclose(template_roYZ_fid);
	FILE * ObjRecon_fid = fopen(ObjRecon_file, "rb");
	if (ObjRecon_fid == NULL)
	{
		cout << ObjRecon_file << " open failed!" << endl;
		system("pause");
		return 0;
	}
	int ObjRecon_size = 200 * 200 * 50;
	float *ObjRecon = new float[ObjRecon_size];
	fread(ObjRecon, sizeof(float), ObjRecon_size, ObjRecon_fid);
	fclose(ObjRecon_fid);

	//cpu
	//计算ObjRecon一个像素在所有波段中的最大值，按照matlab中的矩阵行优先存储
	float *image2D_XY = new float[200 * 200];//按照行优先排列
	double image2D_XY_sum = 0;
	for (int i = 0; i < 200; i++)//行循环
	{
		for (int j = 0; j < 200; j++)//列循环
		{
			image2D_XY[i * 200 + j] = ObjRecon[i * 200 + j];
			for (int b = 0; b < 50; b++)//波段循环
			{
				if (image2D_XY[i * 200 + j] < ObjRecon[b * 200 * 200 + i * 200 + j])
				{
					image2D_XY[i * 200 + j] = ObjRecon[b * 200 * 200 + i * 200 + j];
				}
			}//波段循环
			image2D_XY_sum += image2D_XY[i * 200 + j];
		}
	}
	//// 对投影二值化: 大于mean的取1， 小于等于mean的取0
	//double image2D_XY_mean = image2D_XY_sum / (200 * 200);
	//float *img2DBW_XY = new float[200 * 200]();
	//for (int i = 0; i < 200 * 200; i++)
	//{
	//	if (image2D_XY[i] > image2D_XY_mean)
	//		img2DBW_XY[i] = 1.0;
	//	else
	//		img2DBW_XY[i] = 0.0;
	//}



	// 改CUDA..................
	float *ObjRecon_gpu;
	check1(cudaMalloc((void**)&ObjRecon_gpu, sizeof(float)*ObjRecon_size), "ObjRecon_gpu cudaMalloc Error", __FILE__, __LINE__);
	check(cudaMemcpy(ObjRecon_gpu, ObjRecon, sizeof(float)*ObjRecon_size, cudaMemcpyHostToDevice), "ObjRecon_gpu cudaMemcpy Error");
	float *image2D_XY_gpu;
	check1(cudaMalloc((void**)&image2D_XY_gpu, sizeof(float)* 200 * 200), "image2D_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	dim3 block_1(32, 32, 1);
	dim3 grid_1((200 + block_1.x - 1) / block_1.x, (200 + block_1.y - 1) / block_1.y, 1);
	kernel_1 << <grid_1, block_1 >> > (ObjRecon_gpu, 200, 200, image2D_XY_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_1 Error");

	thrust::device_ptr<float> dev_ptr(image2D_XY_gpu);
	double image2D_XY_mean = thrust::reduce(dev_ptr, dev_ptr + size_t(200 * 200), (float)0, thrust::plus<float>()) / (200 * 200);
	float *img2DBW_XY_gpu;
	check1(cudaMalloc((void**)&img2DBW_XY_gpu, sizeof(float)* 200 * 200), "img2DBW_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	int threadNum_2 = 256;
	int blockNum_2 = (200 * 200 - 1) / threadNum_2 + 1;
	kernel_2 << <blockNum_2, threadNum_2 >> > (image2D_XY_gpu, 200*200, image2D_XY_mean, img2DBW_XY_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_2 Error");

	float *template_roXY_gpu;
	check1(cudaMalloc((void**)&template_roXY_gpu, sizeof(float) * template_roXY_size), "template_roXY_gpu cudaMalloc Error", __FILE__, __LINE__);
	check(cudaMemcpy(template_roXY_gpu, template_roXY, sizeof(float)*template_roXY_size, cudaMemcpyHostToDevice), "template_roXY_gpu cudaMemcpy Error");
	double *err_XY_gpu;
	check1(cudaMalloc((void**)&err_XY_gpu, sizeof(double) * rotationAngleXY_size), "err_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	int threadNum_3 = 256;
	int blockNum_3 = (rotationAngleXY_size - 1) / threadNum_2 + 1;
	kernel_3 << <blockNum_3, threadNum_3 >> > (template_roXY_gpu, img2DBW_XY_gpu, rotationAngleXY_size, err_XY_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_3 Error");

	//求err_XY_gpu的最小值
	double *err_XY = new double[rotationAngleXY_size];
	check(cudaMemcpy(err_XY, err_XY_gpu, sizeof(double)*rotationAngleXY_size, cudaMemcpyDeviceToHost), "err_XY cudaMemcpy Error");
	double err_XY_min = DBL_MAX;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] < err_XY_min)
			err_XY_min = err_XY[i];
	}
	//找到最小值对应的索引
	int idx;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] == err_XY_min)
		{
			idx = i;
			break;
		}
	}
	//第一次旋转
	float *imageRotated3D_gpu;
	check1(cudaMalloc((void**)&imageRotated3D_gpu, sizeof(float) * 200 * 200 * 50), "imageRotated3D_gpu cudaMalloc Error", __FILE__, __LINE__);
	ObjRecon_imrotate3_gpu(ObjRecon_gpu, -rotationAngleXY[idx], imageRotated3D_gpu);

	//求 y-z面的投影,计算imageRotated3D一个像素在行方向的最大值
	float *image2D_YZ_gpu;
	check1(cudaMalloc((void**)&image2D_YZ_gpu, sizeof(float) * 200 * 50), "image2D_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);//200行*50列按照imageRotated3D的列优先排列，是matlab中按照列优先排列
	dim3 block_4(32, 32, 1);
	dim3 grid_4((50 + block_4.x - 1) / block_4.x, (200 + block_4.y - 1) / block_4.y, 1);
	kernel_4 << <grid_4, block_4 >> > (imageRotated3D_gpu, image2D_YZ_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_4 Error");
	//image2D_YZ_gpu求和、均值
	thrust::device_ptr<float> dev_ptr1(image2D_YZ_gpu);
	double image2D_YZ_mean = thrust::reduce(dev_ptr1, dev_ptr1 + size_t(200 * 50), (float)0, thrust::plus<float>()) / (200 * 50) + 14;
	//二值化 y-z面，大于mean的取1， 小于等于mean的取0
	float *img2DBW_YZ_gpu;
	check1(cudaMalloc((void**)&img2DBW_YZ_gpu, sizeof(float) * 200 * 50), "img2DBW_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);
	int threadNum_5 = 256;
	int blockNum_5 = (200 * 50 - 1) / threadNum_5 + 1;
	kernel_5 << <blockNum_5, threadNum_5 >> > (image2D_YZ_gpu, image2D_YZ_mean, img2DBW_YZ_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_5 Error");

	//对每个角度的误差 初始化
	float *template_roYZ_gpu;
	check1(cudaMalloc((void**)&template_roYZ_gpu, sizeof(float) * template_roYZ_size), "template_roYZ_gpu cudaMalloc Error", __FILE__, __LINE__);
	check(cudaMemcpy(template_roYZ_gpu, template_roYZ, sizeof(float)*template_roYZ_size, cudaMemcpyHostToDevice), "template_roYZ_gpu cudaMemcpy Error");
	double *err_YZ_gpu;
	check1(cudaMalloc((void**)&err_YZ_gpu, sizeof(double) * rotationAngleYZ_size), "err_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);
	int threadNum_6 = 256;
	int blockNum_6 = (rotationAngleYZ_size - 1) / threadNum_6 + 1;
	kernel_6 << <blockNum_5, threadNum_5 >> > (template_roYZ_gpu, img2DBW_YZ_gpu, rotationAngleYZ_size, err_YZ_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_6 Error");
	//求err_YZ_gpu的最小值和最小值的索引
	double *err_YZ = new double[rotationAngleYZ_size];
	check(cudaMemcpy(err_YZ, err_YZ_gpu, sizeof(double)*rotationAngleYZ_size, cudaMemcpyDeviceToHost), "err_YZ cudaMemcpy Error");
	double err_YZ_min = DBL_MAX;
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		if (err_YZ[i] < err_YZ_min)
			err_YZ_min = err_YZ[i];
	}
	int idx2;
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		if (err_YZ[i] == err_YZ_min)
		{
			idx2 = i;
			break;
		}
	}
	//imageRotated3D旋转，绕X轴逆旋转rotationAngleYZ(idx2)度
	//先把imageRotated3D_gpu的维度变换一下，列变成波段，波段变成列，行变成反着，(200 * 200 * 50)变成(200行 * 50列 * 200)波段
	float *imageRotated3D_gpu_1;
	check1(cudaMalloc((void**)&imageRotated3D_gpu_1, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_1 cudaMalloc Error", __FILE__, __LINE__);
	dim3 block_7(8, 8, 8);
	dim3 grid_7((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
	kernel_7 << <grid_7, block_7 >> > (imageRotated3D_gpu, imageRotated3D_gpu_1);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_7 Error");
	//第二次旋转
	float *imageRotated3D_gpu_2;
	check1(cudaMalloc((void**)&imageRotated3D_gpu_2, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_2 cudaMalloc Error", __FILE__, __LINE__);
	ObjRecon_imrotate3_X_gpu(imageRotated3D_gpu_1, rotationAngleYZ[idx2], imageRotated3D_gpu_2);
	//再把维度变换成原来的
	dim3 block_8(8, 8, 8);
	dim3 grid_8((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
	kernel_8 << <grid_8, block_8 >> > (imageRotated3D_gpu_2, imageRotated3D_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_8 Error");
	//计算imageRotated3D_gpu的均值
	thrust::device_ptr<float> dev_ptr2(imageRotated3D_gpu);
	double imageRotated3D_x_mean = thrust::reduce(dev_ptr2, dev_ptr2 + size_t(200 * 200 * 50), (float)0, thrust::plus<float>()) / (200 * 200 * 50) + 4;

	check(cudaMemcpy(ObjRecon, imageRotated3D_gpu, sizeof(float)*ObjRecon_size, cudaMemcpyDeviceToHost), "ObjRecon cudaMemcpy Error");

	





	//// 对每个角度的误差 初始化
	//double *err_XY = new double[rotationAngleXY_size];
	//double err_XY_min = DBL_MAX;
	////求二值化结果对于每一个角度的误差,GPU中可以直接并发一次执行这个循环
	//for (int i = 0; i < rotationAngleXY_size; i++)
	//{
	//	//计算两个矩阵的均方误差
	//	double sum_temp = 0;
	//	for (int j = 0; j < 200; j++)//行循环
	//	{
	//		for (int k = 0; k < 200; k++)//列循环
	//		{
	//			sum_temp += (template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k])*(template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k]);
	//		}
	//	}
	//	err_XY[i] = sum_temp / (200 * 200);
	//	if (err_XY[i] < err_XY_min)
	//		err_XY_min = err_XY[i];
	//}
	////找到最小值对应的索引
	//int idx;
	//for (int i = 0; i < rotationAngleXY_size; i++)
	//{
	//	if (err_XY[i] == err_XY_min)
	//	{
	//		idx = i;
	//		break;
	//	}
	//}
	//
	//ObjRecon（200*200*50个float，行优先排列），绕Z轴顺时针旋转rotationAngleXY[idx]度
	//旋转角度为正：逆时针，负：顺时针
	//float *imageRotated3D = ObjRecon_imrotate3(ObjRecon, -rotationAngleXY[idx]);
	////* Y - Z rotation */
	////求 y-z面的投影,计算imageRotated3D一个像素在行方向的最大值
	//float *image2D_YZ = new float[200 * 50];//200行*50列按照imageRotated3D的列优先排列，是matlab中按照列优先排列
	//double image2D_YZ_sum = 0;
	//for (int i = 0; i < 50; i++)//波段循环
	//{
	//	for (int j = 0; j < 200; j++)//行循环
	//	{
	//		image2D_YZ[i * 200 + j] = -FLT_MAX;
	//		for (int k = 0; k < 200; k++)//列循环，求一行的最大值
	//		{
	//			if (image2D_YZ[i * 200 + j] < imageRotated3D[i * 200 * 200 + j * 200 + k])
	//			{
	//				image2D_YZ[i * 200 + j] = imageRotated3D[i * 200 * 200 + j * 200 + k];
	//			}
	//		}
	//		image2D_YZ_sum += image2D_YZ[i * 200 + j];
	//	}
	//}
	//double image2D_YZ_mean = image2D_YZ_sum / (200 * 50) + 14;
	//
	////二值化 y-z面，大于mean的取1， 小于等于mean的取0
	//float *img2DBW_YZ = new float[200 * 50];
	//for (int i = 0; i < 200 * 50; i++)
	//{
	//	if (image2D_YZ[i] > image2D_YZ_mean)
	//		img2DBW_YZ[i] = 1.0;
	//	else
	//		img2DBW_YZ[i] = 0.0;
	//}
	//
	////对每个角度的误差 初始化
	//double *err_YZ = new double[rotationAngleYZ_size];
	//double err_YZ_min = DBL_MAX;
	//求二值化结果对于每一个角度的误差，GPU中可以直接并发一次执行这个循环
	//for (int i = 0; i < rotationAngleYZ_size; i++)
	//{
	//	//计算两个矩阵的均方误差
	//	double sum_temp = 0;
	//	for (int j = 0; j < 200; j++)//行循环
	//	{
	//		for (int k = 0; k < 50; k++)//列循环
	//		{
	//			//template_roYZ是200行*50列*31波段，行优先排列，img2DBW_YZ是列优先排列的
	//			sum_temp += (template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j])*(template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j]);
	//		}
	//	}
	//	err_YZ[i] = sum_temp / (200 * 50);
	//	if (err_YZ[i] < err_YZ_min)
	//		err_YZ_min = err_YZ[i];
	//}
	////找到最小值对应的索引
	//int idx2;
	//for (int i = 0; i < rotationAngleYZ_size; i++)
	//{
	//	if (err_YZ[i] == err_YZ_min)
	//	{
	//		idx2 = i;
	//		break;
	//	}
	//}
	//
	////imageRotated3D旋转，绕X轴逆旋转rotationAngleYZ(idx2)度
	////float *imageRotated3D_x = ObjRecon_imrotate3_X(imageRotated3D, rotationAngleYZ[idx2]);
	//
	//// Crop Out，把旋转完的3D图像中的鱼切出来
	//// 二值化旋转后的图像
	////计算imageRotated3D_x的均值
	//double imageRotated3D_x_sum = 0;
	//for (int i = 0; i < 200 * 200 * 50; i++)
	//{
	//	imageRotated3D_x_sum += imageRotated3D_x[i];
	//}
	//double imageRotated3D_x_mean = imageRotated3D_x_sum / (200 * 200 * 50) + 4;

	int *idx_2 = new int[200 * 200 * 50]();//imageRotated3D_x大于均值的索引
	int idx_2_size = 0;
	for (int i = 0; i < 200 * 200 * 50; i++)
	{
		if (ObjRecon[i] > imageRotated3D_x_mean)
		{
			idx_2_size++;
			idx_2[idx_2_size] = i;
		}
	}
	//idx_2数组的每一个数字转换成imageRotated3D_x（200行*200列*50波段）的行号，列号，波段号
	float *x = new float[idx_2_size]; float x_sum = 0;
	float *y = new float[idx_2_size]; float y_sum = 0;
	float *z = new float[idx_2_size]; float z_sum = 0;
	for (int i = 0; i < idx_2_size; i++)
	{
		z[i] = idx_2[i] / (200 * 200);
		int yushu = idx_2[i] % (200 * 200);
		x[i] = yushu / 200;
		y[i] = yushu % 200;

		x_sum += x[i];
		y_sum += y[i];
		z_sum += z[i];
	}
	int CentroID[3];
	CentroID[0] = int(x_sum / idx_2_size + 0.5);
	CentroID[1] = int(y_sum / idx_2_size + 0.5);
	CentroID[2] = int(z_sum / idx_2_size + 0.5);
	//CentroID数组在matlab中是[89,91,24]，我计算的是[86,91,24],x相差3，是npp旋转和matlab的结果有误差造成的，误差也不大，算正常！！

	// 保留质心坐标周围的区域，坐标索引在matlab的数上要减去1
	// 行范围：【CentroID(0)-61：CentroID(0)+33】 。列范围：【CentroID(2)-38：CentroID(2)+37】。所有的波段
	int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//行
	int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//列
	int	ZObj = 50;//波段
	float *ObjReconRed = new float[XObj*YObj*ZObj];
	for (int i = 0; i < ZObj; i++)//波段循环
	{
		for (int j = 0; j < XObj; j++)//行循环
		{
			for (int k = 0; k < YObj; k++)//列循环
			{
				ObjReconRed[i*XObj*YObj + j*YObj + k] =
					ObjRecon[i * 200 * 200 + (CentroID[0] - 61 + j) * 200 + CentroID[2] - 38 + k];
			}
		}
	}
	// size of reference atlas
	//int XRef = 95; int YRef = 76; int ZRef = 50;

	//不再做matlab中的interp3处理，所以数组ObjReconRed就是最后的结果，即matlab的RescaledRed数组



	auto time_end = system_clock::now();
	auto duration = duration_cast<microseconds>(time_end - time_start);
	float usetime_total = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	cout << "finish，用时：" << usetime_total << endl;
	system("pause");
	return 0;
}



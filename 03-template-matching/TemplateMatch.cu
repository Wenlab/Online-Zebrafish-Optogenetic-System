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
#include <chrono>//��׼ģ�������ʱ���йص�ͷ�ļ�
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>
#include "gdal_mdreader.h"
#include "gdalwarper.h"
#include "ogrsf_frmts.h"

using namespace std;
using namespace chrono;
//ͼ�񼸺α任C++ʵ��--����ƽ�ƣ���ת�����У�����
//https://blog.csdn.net/duiwangxiaomi/article/details/109532590


int clockRate = 1.0;
//��ӡ�豸��Ϣ
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("GPU Parament��\n");
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
//CUDA ��ʼ��
bool InitCUDA()
{
	int count;
	//ȡ��֧��Cuda��װ�õ���Ŀ
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//��ӡ�豸��Ϣ
		printDeviceProp(prop);
		//����Կ���ʱ��Ƶ��
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
	//�÷�check1(cudaMalloc((void**)&aa, 10 * sizeof(float)), "aa cudaMalloc Error", __FILE__, __LINE__);
	if (res != cudaSuccess)
	{
		printf((warningstring + " !\n").c_str());
		printf("   Error text: %s   Error code: %d\n", cudaGetErrorString(res), res);
		printf("   Line:    %d    File:    %s\n", linenum, file);
		system("pause");
		exit(0);
	}
}
//�鿴GPU�����Ƿ���ȷ
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

//npp��ͼ����ת����nppiRotate_32f_C1Rʹ�÷�����
//����ת�Ƕ�Ϊ����ͼ������½������ȿ�ʼ���ģ�˳ʱ����ת�����Ͻǿ�ʼ���ģ���ʱ����ת
//����ת�Ƕ�Ϊ����ͼ������½������ȿ�ʼ���ģ���ʱ����ת�����Ͻǿ�ʼ���ģ�˳ʱ����ת
float *ObjRecon_imrotate3(float *ObjRecon, double nAngle)
{
	//float *input_image = new float[200*200];
	float *imageRotated3D = new float[200 * 200 * 50];
	//for (int i = 0; i < 200 * 200; i++)
	//{
	//	input_image[i] = ObjRecon[i];
	//}

	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 200;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);
	//check(cudaMemcpy(input_image_gpu, input_image, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 50; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	////��תǰ��ĵ�һ�����ηֱ�д��������
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
	////��ͼ������Ͻ�һ��һ�е�д
	//ds1->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, ObjRecon_1, 200, 200, GDT_Float32, 0, 0);
	//ds2->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, imageRotated3D_1, 200, 200, GDT_Float32, 0, 0);
	//GDALClose(ds1);
	//GDALClose(ds2);



	return imageRotated3D;
}

//����X����ת
float *ObjRecon_imrotate3_X(float *imageRotated3D, double nAngle)
{
	//imageRotated3D(200*200*50)ת���ɣ��б�ɲ��Σ����α���У��б�ɷ��ţ����200��*50��*200����
	float *ObjRecon = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//�������ѭ�����������ѭ��
	{
		for (int j = 0; j < 200; j++)//�����ѭ�����������ѭ����������
		{
			for (int k = 0; k < 50; k++)//�����ѭ��������Ĳ���ѭ��
			{
				//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
				ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
			}
		}
	}

	float *imageRotated3D_rotate = new float[200 * 50 * 200];

	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_rotate + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	////��תǰ��ĵ�һ�����ηֱ�д��������
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
	////��ͼ������Ͻ�һ��һ�е�д
	//ds1->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, ObjRecon_1, 200, 200, GDT_Float32, 0, 0);
	//ds2->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, imageRotated3D_1, 200, 200, GDT_Float32, 0, 0);
	//GDALClose(ds1);
	//GDALClose(ds2);

	//�ٱ任��ԭ����ά�ȷֲ�
	//200��*50��*200���Ρ�>>200��*200��*50��
	float *imageRotated3D_rotate_return = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//�������ѭ�����������ѭ��
	{
		for (int j = 0; j < 200; j++)//�����ѭ�����������ѭ����������
		{
			for (int k = 0; k < 50; k++)//�����ѭ��������Ĳ���ѭ��
			{
				imageRotated3D_rotate_return[(49 - k) * 200 * 200 + (199 - j) * 200 + i] = imageRotated3D_rotate[i * 200 * 50 + j * 50 + k];
			}
		}
	}

	return imageRotated3D_rotate_return;
}

//cpu�汾
int main0()
{
	//��ʼ��ʱ
	auto time_start = system_clock::now();
	GDALAllRegister();
	//����֧������·��
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	OGRRegisterAll();
	//CUDA ��ʼ��
	if (!InitCUDA())
	{
		cout << "CUDA֧���µ��Կ��豸��ʼ��ʧ��!" << endl;
		system("pause");
		return 0;
	}

	const char *rotationAngleXY_file = "F:/Archive/rotationAngleXY.dat";//360��double
	const char *rotationAngleYZ_file = "F:/Archive/rotationAngleYZ.dat";//31��double
	const char *template_roXY_file = "F:/Archive/template_roXY.dat";//200*200*360��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	const char *template_roYZ_file = "F:/Archive/template_roYZ.dat";//200*50*31��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	const char *ObjRecon_file = "F:/Archive/ObjRecon.dat";//200*200*50��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������

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

	//����ObjReconһ�����������в����е����ֵ������matlab�еľ��������ȴ洢
	float *image2D_XY = new float[200 * 200];//��������������
	double image2D_XY_sum = 0;
	for (int i = 0; i < 200; i++)//��ѭ��
	{
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			image2D_XY[i * 200 + j] = ObjRecon[i * 200 + j];
			for (int b = 0; b < 50; b++)//����ѭ��
			{
				if (image2D_XY[i * 200 + j] < ObjRecon[b * 200 * 200 + i * 200 + j])
				{
					image2D_XY[i * 200 + j] = ObjRecon[b * 200 * 200 + i * 200 + j];
				}
			}//����ѭ��
			image2D_XY_sum += image2D_XY[i * 200 + j];
		}
	}
	// ��ͶӰ��ֵ��: ����mean��ȡ1�� С�ڵ���mean��ȡ0
	double image2D_XY_mean = image2D_XY_sum / (200 * 200);
	float *img2DBW_XY = new float[200 * 200]();
	for (int i = 0; i < 200*200; i++)
	{
		if (image2D_XY[i] > image2D_XY_mean)
			img2DBW_XY[i] = 1.0;
		else
			img2DBW_XY[i] = 0.0;
	}

	// ��ÿ���Ƕȵ���� ��ʼ��
	double *err_XY = new double[rotationAngleXY_size];
	double err_XY_min = DBL_MAX;
	//���ֵ���������ÿһ���Ƕȵ����,GPU�п���ֱ�Ӳ���һ��ִ�����ѭ��
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 200; k++)//��ѭ��
			{
				sum_temp += (template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k])*(template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k]);
			}
		}
		err_XY[i] = sum_temp / (200 * 200);
		if (err_XY[i] < err_XY_min)
			err_XY_min = err_XY[i];
	}
	//�ҵ���Сֵ��Ӧ������
	int idx;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] == err_XY_min)
		{
			idx = i;
			break;
		}
	}
	
	//ObjRecon��200*200*50��float�����������У�����Z��˳ʱ����תrotationAngleXY[idx]��
	//��ת�Ƕ�Ϊ������ʱ�룬����˳ʱ��
	float *imageRotated3D = ObjRecon_imrotate3(ObjRecon, -rotationAngleXY[idx]);

	/* Y - Z rotation */
	//�� y-z���ͶӰ,����imageRotated3Dһ���������з�������ֵ
	float *image2D_YZ = new float[200 * 50];//200��*50�а���imageRotated3D�����������У���matlab�а�������������
	double image2D_YZ_sum = 0;
	for (int i = 0; i < 50; i++)//����ѭ��
	{
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			image2D_YZ[i * 200 + j] = -FLT_MAX;
			for (int k = 0; k < 200; k++)//��ѭ������һ�е����ֵ
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

	//��ֵ�� y-z�棬����mean��ȡ1�� С�ڵ���mean��ȡ0
	float *img2DBW_YZ = new float[200 * 50];
	for (int i = 0; i < 200 * 50; i++)
	{
		if (image2D_YZ[i] > image2D_YZ_mean)
			img2DBW_YZ[i] = 1.0;
		else
			img2DBW_YZ[i] = 0.0;
	}

	//��ÿ���Ƕȵ���� ��ʼ��
	double *err_YZ = new double[rotationAngleYZ_size];
	double err_YZ_min = DBL_MAX;
	//���ֵ���������ÿһ���Ƕȵ���GPU�п���ֱ�Ӳ���һ��ִ�����ѭ��
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 50; k++)//��ѭ��
			{
				//template_roYZ��200��*50��*31���Σ����������У�img2DBW_YZ�����������е�
				sum_temp += (template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j])*(template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j]);
			}
		}
		err_YZ[i] = sum_temp / (200 * 50);
		if (err_YZ[i] < err_YZ_min)
			err_YZ_min = err_YZ[i];
	}
	//�ҵ���Сֵ��Ӧ������
	int idx2;
	for (int i = 0; i < rotationAngleYZ_size; i++)
	{
		if (err_YZ[i] == err_YZ_min)
		{
			idx2 = i;
			break;
		}
	}

	//imageRotated3D��ת����X������תrotationAngleYZ(idx2)��
	float *imageRotated3D_x = ObjRecon_imrotate3_X(imageRotated3D, rotationAngleYZ[idx2]);

	// Crop Out������ת���3Dͼ���е����г���
	// ��ֵ����ת���ͼ��
	//����imageRotated3D_x�ľ�ֵ
	double imageRotated3D_x_sum = 0;
	for (int i = 0; i < 200*200*50; i++)
	{
		imageRotated3D_x_sum += imageRotated3D_x[i];
	}
	double imageRotated3D_x_mean = imageRotated3D_x_sum / (200 * 200 * 50) + 4;

	int *BWObjRecon = new int[200 * 200 * 50]();
	int *idx_2 = new int[200 * 200 * 50]();//imageRotated3D_x���ھ�ֵ������
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
	//idx_2�����ÿһ������ת����BWObjRecon��200��*200��*50���Σ����кţ��кţ����κ�
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
	//CentroID������matlab����[89,91,24]���Ҽ������[86,91,24],x���3����npp��ת��matlab�Ľ���������ɵģ����Ҳ��������������

	// ��������������Χ����������������matlab������Ҫ��ȥ1
	// �з�Χ����CentroID(0)-61��CentroID(0)+33�� ���з�Χ����CentroID(2)-38��CentroID(2)+37�������еĲ���
	int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//��
	int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//��
	int	ZObj = 50;//����
	float *ObjReconRed = new float[XObj*YObj*ZObj];
	for (int i = 0; i < ZObj; i++)//����ѭ��
	{
		for (int j = 0; j < XObj; j++)//��ѭ��
		{
			for (int k = 0; k < YObj; k++)//��ѭ��
			{
				ObjReconRed[i*XObj*YObj + j*YObj + k] =
					imageRotated3D_x[i*200*200 + (CentroID[0] - 61 + j)*200 + CentroID[2] - 38 + k];
			}
		}
	}
	// size of reference atlas
	//int XRef = 95; int YRef = 76; int ZRef = 50;

	//������matlab�е�interp3������������ObjReconRed�������Ľ������matlab��RescaledRed����



	auto time_end = system_clock::now();
	auto duration = duration_cast<microseconds>(time_end - time_start);
	float usetime_total = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	cout << "finish����ʱ��" << usetime_total << endl;
	system("pause");
    return 0;
}


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
void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu)
{
	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 200;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 50; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon_gpu + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_gpu + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
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
//����X����ת
void ObjRecon_imrotate3_X_gpu(float *imageRotated3D_gpu_1, double nAngle, float *imageRotated3D_gpu_2)
{
	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, imageRotated3D_gpu_1 + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_gpu_2 + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
	}
}
//�ٱ任��ԭ����ά��
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


//gpu�汾
int main()
{
	//��ʼ��ʱ
	auto time_start = system_clock::now();
	GDALAllRegister();
	//����֧������·��
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	OGRRegisterAll();
	//CUDA ��ʼ��
	if (!InitCUDA())
	{
		cout << "CUDA֧���µ��Կ��豸��ʼ��ʧ��!" << endl;
		system("pause");
		return 0;
	}

	const char *rotationAngleXY_file = "F:/Archive/rotationAngleXY.dat";//360��double
	const char *rotationAngleYZ_file = "F:/Archive/rotationAngleYZ.dat";//31��double
	const char *template_roXY_file = "F:/Archive/template_roXY.dat";//200*200*360��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	const char *template_roYZ_file = "F:/Archive/template_roYZ.dat";//200*50*31��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	const char *ObjRecon_file = "F:/Archive/ObjRecon.dat";//200*200*50��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������

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
	//����ObjReconһ�����������в����е����ֵ������matlab�еľ��������ȴ洢
	float *image2D_XY = new float[200 * 200];//��������������
	double image2D_XY_sum = 0;
	for (int i = 0; i < 200; i++)//��ѭ��
	{
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			image2D_XY[i * 200 + j] = ObjRecon[i * 200 + j];
			for (int b = 0; b < 50; b++)//����ѭ��
			{
				if (image2D_XY[i * 200 + j] < ObjRecon[b * 200 * 200 + i * 200 + j])
				{
					image2D_XY[i * 200 + j] = ObjRecon[b * 200 * 200 + i * 200 + j];
				}
			}//����ѭ��
			image2D_XY_sum += image2D_XY[i * 200 + j];
		}
	}
	//// ��ͶӰ��ֵ��: ����mean��ȡ1�� С�ڵ���mean��ȡ0
	//double image2D_XY_mean = image2D_XY_sum / (200 * 200);
	//float *img2DBW_XY = new float[200 * 200]();
	//for (int i = 0; i < 200 * 200; i++)
	//{
	//	if (image2D_XY[i] > image2D_XY_mean)
	//		img2DBW_XY[i] = 1.0;
	//	else
	//		img2DBW_XY[i] = 0.0;
	//}



	// ��CUDA..................
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

	//��err_XY_gpu����Сֵ
	double *err_XY = new double[rotationAngleXY_size];
	check(cudaMemcpy(err_XY, err_XY_gpu, sizeof(double)*rotationAngleXY_size, cudaMemcpyDeviceToHost), "err_XY cudaMemcpy Error");
	double err_XY_min = DBL_MAX;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] < err_XY_min)
			err_XY_min = err_XY[i];
	}
	//�ҵ���Сֵ��Ӧ������
	int idx;
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		if (err_XY[i] == err_XY_min)
		{
			idx = i;
			break;
		}
	}
	//��һ����ת
	float *imageRotated3D_gpu;
	check1(cudaMalloc((void**)&imageRotated3D_gpu, sizeof(float) * 200 * 200 * 50), "imageRotated3D_gpu cudaMalloc Error", __FILE__, __LINE__);
	ObjRecon_imrotate3_gpu(ObjRecon_gpu, -rotationAngleXY[idx], imageRotated3D_gpu);

	//�� y-z���ͶӰ,����imageRotated3Dһ���������з�������ֵ
	float *image2D_YZ_gpu;
	check1(cudaMalloc((void**)&image2D_YZ_gpu, sizeof(float) * 200 * 50), "image2D_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);//200��*50�а���imageRotated3D�����������У���matlab�а�������������
	dim3 block_4(32, 32, 1);
	dim3 grid_4((50 + block_4.x - 1) / block_4.x, (200 + block_4.y - 1) / block_4.y, 1);
	kernel_4 << <grid_4, block_4 >> > (imageRotated3D_gpu, image2D_YZ_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_4 Error");
	//image2D_YZ_gpu��͡���ֵ
	thrust::device_ptr<float> dev_ptr1(image2D_YZ_gpu);
	double image2D_YZ_mean = thrust::reduce(dev_ptr1, dev_ptr1 + size_t(200 * 50), (float)0, thrust::plus<float>()) / (200 * 50) + 14;
	//��ֵ�� y-z�棬����mean��ȡ1�� С�ڵ���mean��ȡ0
	float *img2DBW_YZ_gpu;
	check1(cudaMalloc((void**)&img2DBW_YZ_gpu, sizeof(float) * 200 * 50), "img2DBW_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);
	int threadNum_5 = 256;
	int blockNum_5 = (200 * 50 - 1) / threadNum_5 + 1;
	kernel_5 << <blockNum_5, threadNum_5 >> > (image2D_YZ_gpu, image2D_YZ_mean, img2DBW_YZ_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_5 Error");

	//��ÿ���Ƕȵ���� ��ʼ��
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
	//��err_YZ_gpu����Сֵ����Сֵ������
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
	//imageRotated3D��ת����X������תrotationAngleYZ(idx2)��
	//�Ȱ�imageRotated3D_gpu��ά�ȱ任һ�£��б�ɲ��Σ����α���У��б�ɷ��ţ�(200 * 200 * 50)���(200�� * 50�� * 200)����
	float *imageRotated3D_gpu_1;
	check1(cudaMalloc((void**)&imageRotated3D_gpu_1, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_1 cudaMalloc Error", __FILE__, __LINE__);
	dim3 block_7(8, 8, 8);
	dim3 grid_7((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
	kernel_7 << <grid_7, block_7 >> > (imageRotated3D_gpu, imageRotated3D_gpu_1);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_7 Error");
	//�ڶ�����ת
	float *imageRotated3D_gpu_2;
	check1(cudaMalloc((void**)&imageRotated3D_gpu_2, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_2 cudaMalloc Error", __FILE__, __LINE__);
	ObjRecon_imrotate3_X_gpu(imageRotated3D_gpu_1, rotationAngleYZ[idx2], imageRotated3D_gpu_2);
	//�ٰ�ά�ȱ任��ԭ����
	dim3 block_8(8, 8, 8);
	dim3 grid_8((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
	kernel_8 << <grid_8, block_8 >> > (imageRotated3D_gpu_2, imageRotated3D_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_8 Error");
	//����imageRotated3D_gpu�ľ�ֵ
	thrust::device_ptr<float> dev_ptr2(imageRotated3D_gpu);
	double imageRotated3D_x_mean = thrust::reduce(dev_ptr2, dev_ptr2 + size_t(200 * 200 * 50), (float)0, thrust::plus<float>()) / (200 * 200 * 50) + 4;

	check(cudaMemcpy(ObjRecon, imageRotated3D_gpu, sizeof(float)*ObjRecon_size, cudaMemcpyDeviceToHost), "ObjRecon cudaMemcpy Error");

	





	//// ��ÿ���Ƕȵ���� ��ʼ��
	//double *err_XY = new double[rotationAngleXY_size];
	//double err_XY_min = DBL_MAX;
	////���ֵ���������ÿһ���Ƕȵ����,GPU�п���ֱ�Ӳ���һ��ִ�����ѭ��
	//for (int i = 0; i < rotationAngleXY_size; i++)
	//{
	//	//������������ľ������
	//	double sum_temp = 0;
	//	for (int j = 0; j < 200; j++)//��ѭ��
	//	{
	//		for (int k = 0; k < 200; k++)//��ѭ��
	//		{
	//			sum_temp += (template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k])*(template_roXY[i * 200 * 200 + j * 200 + k] - img2DBW_XY[j * 200 + k]);
	//		}
	//	}
	//	err_XY[i] = sum_temp / (200 * 200);
	//	if (err_XY[i] < err_XY_min)
	//		err_XY_min = err_XY[i];
	//}
	////�ҵ���Сֵ��Ӧ������
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
	//ObjRecon��200*200*50��float�����������У�����Z��˳ʱ����תrotationAngleXY[idx]��
	//��ת�Ƕ�Ϊ������ʱ�룬����˳ʱ��
	//float *imageRotated3D = ObjRecon_imrotate3(ObjRecon, -rotationAngleXY[idx]);
	////* Y - Z rotation */
	////�� y-z���ͶӰ,����imageRotated3Dһ���������з�������ֵ
	//float *image2D_YZ = new float[200 * 50];//200��*50�а���imageRotated3D�����������У���matlab�а�������������
	//double image2D_YZ_sum = 0;
	//for (int i = 0; i < 50; i++)//����ѭ��
	//{
	//	for (int j = 0; j < 200; j++)//��ѭ��
	//	{
	//		image2D_YZ[i * 200 + j] = -FLT_MAX;
	//		for (int k = 0; k < 200; k++)//��ѭ������һ�е����ֵ
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
	////��ֵ�� y-z�棬����mean��ȡ1�� С�ڵ���mean��ȡ0
	//float *img2DBW_YZ = new float[200 * 50];
	//for (int i = 0; i < 200 * 50; i++)
	//{
	//	if (image2D_YZ[i] > image2D_YZ_mean)
	//		img2DBW_YZ[i] = 1.0;
	//	else
	//		img2DBW_YZ[i] = 0.0;
	//}
	//
	////��ÿ���Ƕȵ���� ��ʼ��
	//double *err_YZ = new double[rotationAngleYZ_size];
	//double err_YZ_min = DBL_MAX;
	//���ֵ���������ÿһ���Ƕȵ���GPU�п���ֱ�Ӳ���һ��ִ�����ѭ��
	//for (int i = 0; i < rotationAngleYZ_size; i++)
	//{
	//	//������������ľ������
	//	double sum_temp = 0;
	//	for (int j = 0; j < 200; j++)//��ѭ��
	//	{
	//		for (int k = 0; k < 50; k++)//��ѭ��
	//		{
	//			//template_roYZ��200��*50��*31���Σ����������У�img2DBW_YZ�����������е�
	//			sum_temp += (template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j])*(template_roYZ[i * 200 * 50 + j * 50 + k] - img2DBW_YZ[k * 200 + j]);
	//		}
	//	}
	//	err_YZ[i] = sum_temp / (200 * 50);
	//	if (err_YZ[i] < err_YZ_min)
	//		err_YZ_min = err_YZ[i];
	//}
	////�ҵ���Сֵ��Ӧ������
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
	////imageRotated3D��ת����X������תrotationAngleYZ(idx2)��
	////float *imageRotated3D_x = ObjRecon_imrotate3_X(imageRotated3D, rotationAngleYZ[idx2]);
	//
	//// Crop Out������ת���3Dͼ���е����г���
	//// ��ֵ����ת���ͼ��
	////����imageRotated3D_x�ľ�ֵ
	//double imageRotated3D_x_sum = 0;
	//for (int i = 0; i < 200 * 200 * 50; i++)
	//{
	//	imageRotated3D_x_sum += imageRotated3D_x[i];
	//}
	//double imageRotated3D_x_mean = imageRotated3D_x_sum / (200 * 200 * 50) + 4;

	int *idx_2 = new int[200 * 200 * 50]();//imageRotated3D_x���ھ�ֵ������
	int idx_2_size = 0;
	for (int i = 0; i < 200 * 200 * 50; i++)
	{
		if (ObjRecon[i] > imageRotated3D_x_mean)
		{
			idx_2_size++;
			idx_2[idx_2_size] = i;
		}
	}
	//idx_2�����ÿһ������ת����imageRotated3D_x��200��*200��*50���Σ����кţ��кţ����κ�
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
	//CentroID������matlab����[89,91,24]���Ҽ������[86,91,24],x���3����npp��ת��matlab�Ľ���������ɵģ����Ҳ��������������

	// ��������������Χ����������������matlab������Ҫ��ȥ1
	// �з�Χ����CentroID(0)-61��CentroID(0)+33�� ���з�Χ����CentroID(2)-38��CentroID(2)+37�������еĲ���
	int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//��
	int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//��
	int	ZObj = 50;//����
	float *ObjReconRed = new float[XObj*YObj*ZObj];
	for (int i = 0; i < ZObj; i++)//����ѭ��
	{
		for (int j = 0; j < XObj; j++)//��ѭ��
		{
			for (int k = 0; k < YObj; k++)//��ѭ��
			{
				ObjReconRed[i*XObj*YObj + j*YObj + k] =
					ObjRecon[i * 200 * 200 + (CentroID[0] - 61 + j) * 200 + CentroID[2] - 38 + k];
			}
		}
	}
	// size of reference atlas
	//int XRef = 95; int YRef = 76; int ZRef = 50;

	//������matlab�е�interp3������������ObjReconRed�������Ľ������matlab��RescaledRed����



	auto time_end = system_clock::now();
	auto duration = duration_cast<microseconds>(time_end - time_start);
	float usetime_total = double(duration.count()) * microseconds::period::num / microseconds::period::den;
	cout << "finish����ʱ��" << usetime_total << endl;
	system("pause");
	return 0;
}



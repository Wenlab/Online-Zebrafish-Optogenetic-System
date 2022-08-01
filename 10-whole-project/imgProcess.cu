#include"imgProcess.h"
#include"kexinLibs.h"
#include"initANDcheck.h"

#include "reconstructionCUDA.cuh"
#include "templateMatchingCUDA.cuh"

#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>

#include <iomanip>
#include <locale>
#include <sstream>
#include <string>


//#include <chrono>//��׼ģ�������ʱ���йص�ͷ�ļ�
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

//#include <assert.h>
//#include <stdio.h>
//#include <stdlib.h>

#include<iostream>
//#include<string>

using namespace std;
//�ع�����   ��Ҫ����  ����ֱ�����ඨ�������ʼ�����Ժ󵥶�дһ����ʼ������
int clockRate = 1.0;
float scale = 1.0 / 4;
int ItN = 10; // ��������
int ROISize = 100;// ROI ��С ���ù�
int BkgMean = 140;// ������ֵ
int SNR = 200;// SNR���ù�
int NxyExt = 0;//�������ķ�Χ����������0�����������
int PSF_size_1 = 512;
int PSF_size_2 = 512;
int PSF_size_3 = 50;
int Nxy = PSF_size_1 + NxyExt * 2; // �Ǹ����� ����ֱ�Ӵ��Դ���
int Nz = PSF_size_3; // ����ֱ���ó��� ���Դ���  Nz = 50

int threadNum_123 = 256;
int blockNum_123 = (PSF_size_1*PSF_size_2*PSF_size_3 - 1) / threadNum_123 + 1;
int threadNum_12 = 256;
int blockNum_12 = (PSF_size_1*PSF_size_2 - 1) / threadNum_12 + 1;
int threadNum_ROI = 256;
int blockNum_ROI = (ROISize * 2 * ROISize * 2 * Nz - 1) / threadNum_ROI + 1;
dim3 block(8, 8, 8);
dim3 grid((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, (PSF_size_3 + block.z - 1) / block.z);
dim3 block_sum(32, 32, 1);
dim3 grid_sum((PSF_size_1 + block.x - 1) / block.x, (PSF_size_2 + block.y - 1) / block.y, 1);

int ObjRecon_size = 200 * 200 * 50;


bool Contour_Area(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2)
{
	return cv::contourArea(contour1) > cv::contourArea(contour2);
}

void FishImageProcess::initialize()
{



	//��ȡPSF��δ�ع����ļ�
	string PSF_1_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/PSF_1_zhuanzhi_float.dat";//matlab�б��������float����
	string X31_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/r20210924_2_X31_resize.tif";
	//��ȡ�Ƕȡ���άģ����Ϣ
	string rotationAngleXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleXY.dat";//360��double
	string rotationAngleYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleYZ.dat";//31��double
	string template_roXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/templateXY.tif";//200*200*360��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	string template_roYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/template_roYZ.dat";//200*50*31��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
	//��ȡ���ڷ�������fixImage
	string fixImage_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/toAffineWithZBB.tif";


	readPSFfromFile(PSF_1_file);
	readRotationAngleFromFile(rotationAngleXY_file, rotationAngleYZ_file);
	readTemplateFromFile(template_roXY_file, template_roYZ_file);
	readFixImageFromFile(fixImage_file);

	initializeFishReg("anatomyList_4bin.txt");

	prepareGPUMemory();
	processPSF();

	return;
}

void FishImageProcess::readPSFfromFile(std::string filename)
{
	cout << "start read PSF1 from file..." << endl;


	FILE *PSF_1_fid = fopen(filename.data(), "rb");
	if (PSF_1_fid == NULL)
	{
		cout << "PSF_1_file open failed!" << endl;
		system("pause");
		return;
	}

	PSF_1 = new float[PSF_size_1*PSF_size_2*PSF_size_3]();
	fread(PSF_1, sizeof(float), PSF_size_1*PSF_size_2*PSF_size_3, PSF_1_fid);

	cout << "read PSF1 done" << endl;

	return;
}


void FishImageProcess::loadImage(unsigned short* imgbuffer)
{
	Img = imgbuffer;

	return;
}

void FishImageProcess::readImageFromFile(std::string filename)
{
	cout << "read: " << filename << endl;
	//Img = readImgFromFile(filename);
	//ʹ��GDAL��ȡtif��ʹ�õ���matlab�ز����õ�����
	GDALAllRegister(); OGRRegisterAll();
	//����֧������·��
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	GDALDataset* poSrcDS = (GDALDataset*)GDALOpen(filename.data(), GA_ReadOnly);
	if (poSrcDS == NULL)
	{
		cout << "image file open failed!" << endl;
		return;
	}
	int wheight = poSrcDS->GetRasterYSize();//��
	int wwidth = poSrcDS->GetRasterXSize();//��
	int bandNum = poSrcDS->GetRasterCount();//������
	GDALDataType dataType = poSrcDS->GetRasterBand(1)->GetRasterDataType();//����

	Img = new unsigned short[PSF_size_1*PSF_size_2]();   //ͼ�������������
	for (int i = 0; i < bandNum; i++)
	{
		////////////////////////////////////��ȡ��ʼ�У�ʼ�У�������������ָ�룬��������������������
		poSrcDS->GetRasterBand(i + 1)->RasterIO(GF_Read, 0, 0, wwidth, wheight, Img, PSF_size_1, PSF_size_2, dataType, 0, 0);
	}
	GDALClose(poSrcDS);

	//check
	//cout << Img[512 * 256 + 256] << endl;

	//cout << "read image file done" << endl;

	return;
}

void FishImageProcess::readImageFromCamera(std::string filename)
{
	//������
	return;
}

void FishImageProcess::readTemplateFromFile(std::string filenameXY, std::string filenameYZ)
{
	cout << "start read templates...." << endl;
	//FILE * template_roXY_fid = fopen(filenameXY.data(), "rb");
	//if (template_roXY_fid == NULL)
	//{
	//	cout << filenameXY << " open failed!" << endl;
	//	system("pause");
	//	return;
	//}
	int template_roXY_size = 200 * 200 * 360;
	template_roXY = new float[template_roXY_size];
	//fread(template_roXY, sizeof(float), template_roXY_size, template_roXY_fid);
	//fclose(template_roXY_fid);

	template_roXY = readImgFromFile(filenameXY);

	cout << "XY template read successful" << endl;



	FILE * template_roYZ_fid = fopen(filenameYZ.data(), "rb");
	if (template_roYZ_fid == NULL)
	{
		cout << filenameYZ << " open failed!" << endl;
		system("pause");
		return;
	}
	int template_roYZ_size = 200 * 50 * 31;
	template_roYZ = new float[template_roYZ_size];
	fread(template_roYZ, sizeof(float), template_roYZ_size, template_roYZ_fid);
	fclose(template_roYZ_fid);
	cout << "YZ template read successful" << endl;

	cout << "read template done" << endl;

	return;
}

void FishImageProcess::readRotationAngleFromFile(std::string filenameAngleXY, std::string filenameAngleYZ)
{
	cout << "start read rotation angle from file.." << endl;
	FILE * rotationAngleXY_fid = fopen(filenameAngleXY.data(), "rb");
	if (rotationAngleXY_fid == NULL)
	{
		cout << filenameAngleXY << " open failed!" << endl;
		system("pause");
		return;
	}
	int rotationAngleXY_size = 360;
	rotationAngleXY = new double[rotationAngleXY_size];
	fread(rotationAngleXY, sizeof(double), rotationAngleXY_size, rotationAngleXY_fid);
	fclose(rotationAngleXY_fid);
	cout << "read XY rotation angle successfule" << endl;

	FILE * rotationAngleYZ_fid = fopen(filenameAngleYZ.data(), "rb");
	if (rotationAngleYZ_fid == NULL)
	{
		cout << filenameAngleYZ << " open failed!" << endl;
		system("pause");
		return;
	}
	int rotationAngleYZ_size = 31;
	rotationAngleYZ = new double[rotationAngleYZ_size];
	fread(rotationAngleYZ, sizeof(double), rotationAngleYZ_size, rotationAngleYZ_fid);
	fclose(rotationAngleYZ_fid);

	cout << "read YZ rotation angle successfule" << endl;
	cout << "read rotation angle done" << endl;

	return;
}

void FishImageProcess::readFixImageFromFile(std::string filename)
{
	float* fixImage = readImgFromFile(filename);
	//cout << nImgSizeX << "   " << nImgSizeY << "   " << bandcount << endl;
	fixtensor = torch::from_blob(fixImage,
		{int(imgSizeAfterCrop_Z), int(imgSizeAfterCrop_Y), int(imgSizeAfterCrop_X) }).toType(torch::kFloat32);
	fixtensor = normalizeTensor(fixtensor);
	cout << "read fix image and convert to normalize tensor" << endl;
	//torch::Device device(torch::kCUDA);
	fixtensor.to(device);
	cout << "copy fix tensor to CUDA" << endl;

	//���紦��ǰ����ͼ���������ǰ��һ��
	cout << "warm up..." << endl;
	for (int i = 0; i < 10; i++)
	{
		model.forward({ fixtensor.to(device),fixtensor.to(device) }).toTensor();
		cout << i << "  " ;
	}
	cout << "model process done" << endl;

	return;
}

void FishImageProcess::initializeFishReg(std::string filename)
{
	FishReg.initialize(filename);

	vector<float> Fix2ZBBAM{ 0.985154,	0.0184487, -0.00942914,
	-0.0166061,	1.13246, -0.102937,
	0.0196408, -0.0078765,	1.25844,
	0.522241, -6.91866, -11.7296 };
	FishReg.getZBB2FixAffineMatrix(Fix2ZBBAM);

	return;
}


void FishImageProcess::prepareGPUMemory()
{
	/*-------׼�������������ڴ桢�Դ棬fft���--------------------------*/
	cout << "start malloc memory..." << endl;

	const int rank = 2;//ά��
	int n[rank] = { PSF_size_1, PSF_size_2 };//n*m
	int *inembed = n;//���������size
	int istride = 1;//����������������Ϊ1
	int idist = n[0] * n[1];//1��������ڴ��С
	int *onembed = n;//�����һ�������size
	int ostride = 1;//ÿ��DFT������������Ϊ1
	int odist = n[0] * n[1];//�����һ��������ڶ�������ľ��룬�������������Ԫ�صľ���
	int batch = PSF_size_3;//�������������

	//��ʼ�����
	cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

	//�ع�
	check(cudaMalloc((void**)&PSF_1_gpu, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "PSF_1_gpu cudaMalloc Error");
	check(cudaMalloc((void**)&PSF_1_gpu_Complex, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "PSF_1_gpu_Complex cudaMalloc Error");
	check(cudaMalloc((void**)&OTF, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "OTF cudaMalloc Error");
	check(cudaMalloc((void**)&ImgEst, PSF_size_1*PSF_size_2 * sizeof(float)), "ImgEst cudaMalloc Error");
	check(cudaMalloc((void**)&Ratio, PSF_size_1*PSF_size_2 * sizeof(float)), "Ratio cudaMalloc Error");
	check(cudaMalloc((void**)&gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "Ratio cudaMalloc Error");
	check(cudaMalloc((void**)&gpuObjRecROI, ROISize * 2 * ROISize * 2 * PSF_size_3 * sizeof(float)), "gpuObjRecROI cudaMalloc Error");
	check(cudaMalloc((void**)&Img_gpu, PSF_size_1*PSF_size_2 * sizeof(unsigned short)), "Img_gpu cudaMalloc Error");
	check(cudaMalloc((void**)&ImgExp, PSF_size_1*PSF_size_2 * sizeof(float)), "ImgExp cudaMalloc Error");
	check(cudaMalloc((void**)&gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "gpuObjRecon_Complex cudaMalloc Error");
	check(cudaMalloc((void**)&float_temp, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "float_temp cudaMalloc Error");
	check(cudaMalloc((void**)&Ratio_Complex, PSF_size_1*PSF_size_2 * sizeof(cufftComplex)), "Ratio_Complex cudaMalloc Error");
	check(cudaMalloc((void**)&fftRatio, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "fftRatio cudaMalloc Error");


	//crop 512*512 to 200*200
	cpuObjRecon = new float[PSF_size_1*PSF_size_2*PSF_size_3]();
	cpuObjRecon_crop = new float[200 * 200 * PSF_size_3];
	check1(cudaMalloc((void**)&gpuObjRecon_crop, sizeof(float)*ObjRecon_size), "gpuObjRecon_crop cudaMalloc Error", __FILE__, __LINE__);

	//XY��ת
	check1(cudaMalloc((void**)&image2D_XY_gpu, sizeof(float) * 200 * 200), "image2D_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&img2DBW_XY_gpu, sizeof(float) * 200 * 200), "img2DBW_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&template_roXY_gpu, sizeof(float) * template_roXY_size), "template_roXY_gpu cudaMalloc Error", __FILE__, __LINE__);
	check(cudaMemcpy(template_roXY_gpu, template_roXY, sizeof(float)*template_roXY_size, cudaMemcpyHostToDevice), "template_roXY_gpu cudaMemcpy Error");
	check1(cudaMalloc((void**)&err_XY_gpu, sizeof(double) * rotationAngleXY_size), "err_XY_gpu cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&imageRotated3D_gpu, sizeof(float) * ObjRecon_size), "imageRotated3D_gpu cudaMalloc Error", __FILE__, __LINE__);
	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 200;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float) * 200 * 200), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)* 200 * 200), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//crop
	cpuObjRotation_crop = new float[200 * 200 * 50];
	check1(cudaMalloc((void**)&ObjCropRed_gpu, sizeof(float)*imgSizeAfterCrop_X*imgSizeAfterCrop_Y*imgSizeAfterCrop_Z),
		"ObjReconRed_gpu cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&imageRotated2D_XY_GPU, sizeof(float) * 200 * 200 * 1), "imageRotated2D_XY cudaMalloc Error", __FILE__, __LINE__);
	check1(cudaMalloc((void**)&imageRotated2D_XY_BW_GPU, sizeof(float) * 200 * 200 * 1), "imageRotated2D_XY_BW_GPU cudaMalloc Error", __FILE__, __LINE__);
	imageRotated2D_XY_BW_CPU = new float[200 * 200]();



	cout << "prepare memory done" << endl;

	return;
}

void FishImageProcess::processPSF()
{
	cout << "��ʼ���ʹ���PSF...." << endl;
	check(cudaMemcpy(PSF_1_gpu, PSF_1, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float), cudaMemcpyHostToDevice), "PSF_1_gpu cudaMemcpy Error");
	//ת���ɸ������鲿��0
	Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (PSF_1_gpu, PSF_1_gpu_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu Zhuan_Complex_kernel Error");
	//*----ʹ��cufftPlanMany�ķ�������������άfft---------------------*/
	cufftExecC2C(fftplanfwd, PSF_1_gpu_Complex, OTF, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu_Complex cufftExecC2C Error");

	//ImgEst��ֵΪ0��Ratio��ֵΪ1
	initial_kernel_1 << <blockNum_12, threadNum_12 >> > (ImgEst, Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_1 Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");
	//gpuObjRecROI��ֵΪ1
	initial_kernel_3 << <blockNum_ROI, threadNum_ROI >> > (gpuObjRecROI, ROISize * 2 * ROISize * 2 * Nz);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_3 Error");

	cout << "��ʼ���ʹ���PSF....down" << endl;

	return;
}

void FishImageProcess::reconImage()
{
	////*----1��ʹ��cufftPlan2d�ķ������ж�άfft----------*/
	cufftHandle plan;
	//cufftResult res;
	cufftResult res= cufftPlan2d(&plan, PSF_size_1, PSF_size_2, CUFFT_C2C);  
	


	check(cudaMemcpy(Img_gpu, Img, PSF_size_1*PSF_size_2 * sizeof(unsigned short), cudaMemcpyHostToDevice), "Img_gpu cudaMemcpy Error");
	//��ȥ������ֵ���������float���͵�����ImgExp��
	ImgExp_ge << <blockNum_12, threadNum_12 >> > (Img_gpu, BkgMean, ImgExp, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "ImgExp_ge Error");


	//Ratio��gpuObjRecon��Ԫ�ض���ֵ1
	Ratio_fuzhi << <blockNum_12, threadNum_12 >> > (Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "Ratio_fuzhi Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");




	//��ʼѭ������
	for (int i = 0; i < ItN; i++)
	{
		////1��fft2(gpuObjRecon)
		Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (gpuObjRecon, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Zhuan_Complex_kernel Error");
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C Error");

		////2��OTF.*fft2(gpuObjRecon_Complex)���������gpuObjRecon_Complex��
		OTF_mul_gpuObjRecon_Complex << <blockNum_123, threadNum_123 >> > (OTF, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "2��OTF.*fft2(gpuObjRecon_Complex) Error");

		////3��ifft2(OTF.*fft2(gpuObjRecon))����任��Ҫ���������ظ���
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C cufft_inverse Error");
		////4������������������ȷ
		ifft2_divide << <blockNum_123, threadNum_123 >> > (gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex ifft2_divide Error");

		/*----�ڶ���gpuObjRecon_Complex��ʵ����ȷ���鲿����ȷ������Ĵ���ֻ����gpuObjRecon_Complex��ʵ����û�õ��鲿----------*/

		////5��ifftshift + real + max(,0)�����ʵ������float_temp��С��0�ĸ�ֵ0
		ifftshift_real_max << <grid, block >> > (gpuObjRecon_Complex, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "5��gpuObjRecon_Complex ifftshift_real_max Error");

		////6��sum( ,3)���ڵ���ά�ϼ���ͣ�����PSF_size_1��PSF_size_2�еľ���ImgEst
		float_temp_sum << <grid_sum, block_sum >> > (float_temp, ImgEst, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "float_temp_sum Error");

		////7��Tmp=mean(   ImgEst(:)   );
		thrust::device_ptr<float> dev_ptr(ImgEst);
		float Tmp = thrust::reduce(dev_ptr, dev_ptr + size_t(PSF_size_1*PSF_size_2), (float)0, thrust::plus<float>()) / (PSF_size_1*PSF_size_2);

		////8��Ratio(1:end,1:end)=ImgExp(1:end,1:end)./(ImgEst(1:end,1:end)+Tmp/SNR)����ת�ɸ��������鲿Ϊ��;
		Ratio_Complex_ge << <blockNum_12, threadNum_12 >> > (ImgExp, ImgEst, Tmp, SNR, Ratio_Complex, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Ratio_Complex_ge Error");

		////9��fft2(Ratio)
		res = cufftExecC2C(plan, Ratio_Complex, Ratio_Complex, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS)
		{
			cout << "Ratio_Complex cufftExecC2C error:" << res << endl;
			system("pause");
			return;
		}

		////10��repmat����ֵNz�飬Ratio_Complex�����ά��fftRatio
		fftRatio_ge << <grid, block >> > (Ratio_Complex, fftRatio, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_ge Error");


		////11��fftRatio.*conj(OTF)���浽fftRatio��
		fftRatio_mul_conjOTF << <blockNum_123, threadNum_123 >> > (fftRatio, OTF, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_mul_conjOTF Error");


		////12��ifft2(       fftRatio.*conj(OTF)       )�������������ظ���
		cufftExecC2C(fftplanfwd, fftRatio, fftRatio, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio cufftExecC2C Error");
		ifft2_divide << <blockNum_123, threadNum_123 >> > (fftRatio, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio ifft2_divide Error");

		////13��max(   real(   ifftshift(   ifftshift(     1),   2)   ),   0);
		ifftshift_real_max << <grid, block >> > (fftRatio, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "13��fftRatio ifftshift_real_max Error");

		////14��gpuObjRecon = gpuObjRecon.*max(  )
		real_multiply << <blockNum_123, threadNum_123 >> > (gpuObjRecon, float_temp, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon real_multiply Error");

	}

	cufftDestroy(plan);   //�����һ�䣬cufftPlan2d���ٲ����Դ�й¶

	if (DEBUG)
	{
		cout << "�ع����" << endl;
	}
	return;
}

void FishImageProcess::cropReconImage()
{
	//������ϣ�ȡֵ����cpuObjRecon
	check(cudaMemcpy(cpuObjRecon, gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float), cudaMemcpyDeviceToHost), "gpuObjRecon to cpuObjRecon cudaMemcpy Error");
	/*  ���crop����CPU����ɵģ��ĳ���GPU�����   */
	//gpuObjRecon crop ��200*200*50
	//����CPU��crop�󴫵�GPU
	////matlab���Ǵ�157��356�У��ܹ�356-157+1=200�С�157-356�У��ܹ�356-157+1=200�С�
	int line_start = Nxy / 2 - ROISize; int line_end = Nxy / 2 + ROISize - 1; int line_total = line_end - line_start + 1;
	int col_start = Nxy / 2 - ROISize; 	int col_end = Nxy / 2 + ROISize - 1; int col_total = col_end - col_start + 1;
	if (DEBUG)
	{
		cout << "line_start: " << line_start << endl;
		cout << "line_end: " << line_end << endl;
		cout << "line_total: " << line_total << endl;
		cout << "col_start: " << col_start << endl;
		cout << "col_end: " << col_end << endl;
		cout << "col_total: " << col_total << endl;
	}

	for (int band = 0; band < PSF_size_3; band++)
	{
		for (int i = 0; i < line_total; i++)//��ѭ��
		{
			for (int j = 0; j < col_total; j++)//��ѭ��
			{
				cpuObjRecon_crop[band * 200 * 200 + i * 200 + j] = cpuObjRecon[band*PSF_size_1*PSF_size_2 + (i + line_start)*PSF_size_2 + j + col_start];
			}
		}
	}
	//float *gpuObjRecon_crop;   //�洢crop���ObjRecon
	check(cudaMemcpy(gpuObjRecon_crop, cpuObjRecon_crop, sizeof(float)*ObjRecon_size, cudaMemcpyHostToDevice), "gpuObjRecon_crop cudaMemcpy Error");

	if (DEBUG)
	{
		//cropReconImage_kernel << <blockNum_123, threadNum_123 >> > (gpuObjRecon, gpuObjRecon_crop);
		cout << "crop�ع��������ݲ�copy��GPU" << endl;
	}
	return;
}


void FishImageProcess::matchingANDrotationXY()
{
	///*   XYƽ���ģ��ƥ�����ת   */
	//if (DEBUG)
	//{
	//	cout << "start XY 2D template matching..." << endl;
	//}
	////GET MIP
	dim3 block_1(32, 32, 1);
	dim3 grid_1((200 + block_1.x - 1) / block_1.x, (200 + block_1.y - 1) / block_1.y, 1);
	kernel_1 << <grid_1, block_1 >> > (gpuObjRecon_crop, 200, 200, image2D_XY_gpu);   
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_1 Error");

	//thrust::device_ptr<float> dev_ptr(image2D_XY_gpu);
	//double image2D_XY_mean = thrust::reduce(dev_ptr, dev_ptr + size_t(200 * 200), (float)0, thrust::plus<float>()) / (200 * 200);
	//if (DEBUG)
	//{
	//	cout << "image2D_XY_mean: " << image2D_XY_mean << endl;
	//}

	//int threadNum_2 = 256;
	//int blockNum_2 = (200 * 200 - 1) / threadNum_2 + 1;
	//kernel_2 << <blockNum_2, threadNum_2 >> > (image2D_XY_gpu, 200 * 200, image2D_XY_mean, img2DBW_XY_gpu);
	//cudaDeviceSynchronize();
	//checkGPUStatus(cudaGetLastError(), "kernel_2 Error");

	//int threadNum_3 = 256;
	//int blockNum_3 = (rotationAngleXY_size - 1) / threadNum_2 + 1;
	//kernel_3 << <blockNum_3, threadNum_3 >> > (template_roXY_gpu, img2DBW_XY_gpu, rotationAngleXY_size, err_XY_gpu);
	//cudaDeviceSynchronize();
	//checkGPUStatus(cudaGetLastError(), "kernel_3 Error");


	////��err_XY_gpu����Сֵ
	//double *err_XY = new double[rotationAngleXY_size];
	//check(cudaMemcpy(err_XY, err_XY_gpu, sizeof(double)*rotationAngleXY_size, cudaMemcpyDeviceToHost), "err_XY cudaMemcpy Error");
	//double err_XY_min = DBL_MAX;
	//int idx;  //�ҵ���Сֵ��Ӧ������
	//for (int i = 0; i < rotationAngleXY_size; i++)
	//{
	//	//cout << i << "   " << err_XY[i] << endl;
	//	if (err_XY[i] < err_XY_min)
	//	{
	//		err_XY_min = err_XY[i];
	//		idx = i;
	//	}
	//}
	////cout << "err_XY_min: " << err_XY_min << endl;
	////cout << "rotation XY idx: " << idx << endl;
	////��һ����ת
	//rotationAngleX = -rotationAngleXY[idx];
	//rotationAngleY = 0;
	//rotationAngleX = 166;
	ObjRecon_imrotate3_gpu(gpuObjRecon_crop, -rotationAngleX, imageRotated3D_gpu);

	//std::cout << "imrotate3 rotationAngleX: " << -rotationAngleX << std::endl;

	if (DEBUG)
	{
		cout << "XY 2D templaet matching and rotation done" << endl;
	}

	return;
}

void FishImageProcess::ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu)
{
	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 200;

	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���

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
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);


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

	return;
}




void FishImageProcess::cropRotatedImage()
{
	if (DEBUG)
	{
		cout << "start crop rotation image..." << endl;
	}
	//	//����imageRotated3D_gpu�ľ�ֵ
	thrust::device_ptr<float> dev_ptr2(imageRotated3D_gpu);
	double imageRotated3D_x_mean = thrust::reduce(dev_ptr2, dev_ptr2 + size_t(ObjRecon_size), (float)0, thrust::plus<float>()) / (ObjRecon_size)+4;

	check(cudaMemcpy(cpuObjRotation_crop, imageRotated3D_gpu, sizeof(float)*ObjRecon_size, cudaMemcpyDeviceToHost), "ObjRecon cudaMemcpy Error");

	////crop
	//int *idx_2 = new int[ObjRecon_size]();//imageRotated3D_x���ھ�ֵ������
	//int idx_2_size = 0;
	//for (int i = 0; i < ObjRecon_size; i++)
	//{
	//	if (cpuObjRotation_crop[i] > imageRotated3D_x_mean)
	//	{
	//		idx_2_size++;
	//		idx_2[idx_2_size] = i;
	//	}
	//}
	////idx_2�����ÿһ������ת����imageRotated3D_x��200��*200��*50���Σ����кţ��кţ����κ�
	//float *x = new float[idx_2_size]; float x_sum = 0;
	//float *y = new float[idx_2_size]; float y_sum = 0;
	//float *z = new float[idx_2_size]; float z_sum = 0;
	//for (int i = 0; i < idx_2_size; i++)
	//{
	//	z[i] = idx_2[i] / (200 * 200);
	//	int yushu = idx_2[i] % (200 * 200);
	//	x[i] = yushu / 200;
	//	y[i] = yushu % 200;

	//	x_sum += x[i];
	//	y_sum += y[i];
	//	z_sum += z[i];
	//}
	//int CentroID[3];
	//CentroID[0] = int(x_sum / idx_2_size + 0.5);
	//CentroID[1] = int(y_sum / idx_2_size + 0.5);
	//CentroID[2] = int(z_sum / idx_2_size + 0.5);
	//if (DEBUG)
	//{
	//	cout << "CentroID: " << CentroID[0] << "   " << CentroID[1] << "  " << CentroID[2] << endl;
	//}
	////CentroID������matlab����[89,91,24]���Ҽ������[86,91,24],x���3����npp��ת��matlab�Ľ���������ɵģ����Ҳ��������������

	//// ��������������Χ����������������matlab������Ҫ��ȥ1
	//// �з�Χ����CentroID(0)-61��CentroID(0)+33�� ���з�Χ����CentroID(2)-38��CentroID(2)+37�������еĲ���
	////int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//��
	////int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//��
	////int	ZObj = 50;//����

	//if (CentroID[0] < 61 || CentroID[1] < 38 || CentroID[0]>167 || CentroID[1]>163)
	//{
	//	cout << "centroID error!!!" << endl;
	//	return;
	//}

	//cropPoint = cv::Point3d(CentroID[0] - 61, CentroID[1] - 38, 0);




	
	dim3 block_1(32, 32, 1);
	dim3 grid_1((200 + block_1.x - 1) / block_1.x, (200 + block_1.y - 1) / block_1.y, 1);
	kernel_1 << <grid_1, block_1 >> > (imageRotated3D_gpu, 200, 200, imageRotated2D_XY_GPU);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_1 Error");

	thrust::device_ptr<float> dev_ptr(imageRotated2D_XY_GPU);
	double imageRotated2D_XY_mean = thrust::reduce(dev_ptr, dev_ptr + size_t(200 * 200), (float)0, thrust::plus<float>()) / (200 * 200);

	cout << "imageRotated2D_XY_mean: " << imageRotated2D_XY_mean << endl;

	
	int threadNum_2 = 256;
	int blockNum_2 = (200 * 200 - 1) / threadNum_2 + 1;
	kernel_2 << <blockNum_2, threadNum_2 >> > (imageRotated2D_XY_GPU, 200 * 200, imageRotated2D_XY_mean, imageRotated2D_XY_BW_GPU);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_2 Error");

	check(cudaMemcpy(imageRotated2D_XY_BW_CPU, imageRotated2D_XY_BW_GPU, sizeof(float) * 200 * 200, cudaMemcpyDeviceToHost), "ObjRecon cudaMemcpy Error");

	cv::Mat temp(200, 200, CV_32FC1, imageRotated2D_XY_BW_CPU);
	cv::Mat temp2 = temp.clone();
	temp.convertTo(temp, CV_8UC1);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(temp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));

	if (contours.size() == 0)
	{
		cout << "no fish detect...." << endl;
		return;
	}

	cv::Rect rect;
	sort(contours.begin(), contours.end(), Contour_Area);
	rect = cv::boundingRect(contours[0]);
	cv::rectangle(temp2, rect, cv::Scalar(128), 2);


	int CentroID[3];
	CentroID[0] = rect.tl().x;
	CentroID[1] = rect.tl().y;
	CentroID[2] = 0;

	cout << "CentroID[0]:" << CentroID[0] << "  CentroID[1]:" << CentroID[1] << " CentroID[2]:" << CentroID[2] << endl;

	if (CentroID[0] + 95 > 200 || CentroID[1] + 76 > 200 || CentroID[0] - 10 < 0)
	{
		cout << "centroID error!!!" << endl;
		return;
	}

	cropPoint = cv::Point3d(CentroID[0] - 10, CentroID[1], 0);


	dim3 block_10(8, 8, 8);
	dim3 grid_10((imgSizeAfterCrop_X + block_10.x - 1) / block_10.x, (imgSizeAfterCrop_Y + block_10.y - 1) / block_10.y, (imgSizeAfterCrop_Z + block_10.z - 1) / block_10.z);
	//__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2)
	//kernel_10 << <grid_10, block_10 >> > (imageRotated3D_gpu, ObjCropRed_gpu, imgSizeAfterCrop_X, imgSizeAfterCrop_Y, imgSizeAfterCrop_Z, CentroID[0], CentroID[1]);
	kernel_11 << <grid_10, block_10 >> > (imageRotated3D_gpu, ObjCropRed_gpu, imgSizeAfterCrop_X, imgSizeAfterCrop_Y, imgSizeAfterCrop_Z, CentroID[0] - 10, CentroID[1]);

	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_10 Error");

	//delete[] idx_2;
	//delete[] x;
	//delete[] y;
	//delete[] z;

	if (DEBUG)
	{
		cout << "crop���" << endl;
	}

	return;
}


void FishImageProcess::libtorchModelProcess()
{
	//libtorch
	//convert image to tensor
	torch::Tensor movingtensor;


	//float* temp1 = new float[50 * 95 * 76]();
	//cudaMemcpy(temp1, ObjCropRed_gpu, sizeof(float) * 50 * 95 * 76, cudaMemcpyDeviceToHost);
	//movingtensor = torch::from_blob(temp1,
	//{ int(imgSizeAfterCrop_Z), int(imgSizeAfterCrop_Y), int(imgSizeAfterCrop_X) }).toType(torch::kFloat32);

	movingtensor = torch::from_blob(ObjCropRed_gpu,
		{ int(imgSizeAfterCrop_Z), int(imgSizeAfterCrop_Y), int(imgSizeAfterCrop_X) }, torch::kCUDA).toType(torch::kFloat32);
	movingtensor = normalizeTensor(movingtensor);
	if (DEBUG)
	{
		cout << movingtensor.sizes() << endl;
		cout << fixtensor.sizes() << endl;
	}
	//cout << "1111" << endl;
	auto output = model.forward({ movingtensor.to(device),fixtensor.to(device) }).toTensor();
	//auto output = model.forward({ movingtensor,fixtensor }).toTensor();
	//cout << "2222" << endl;
	Moving2FixAM = rescaleAffineMatrix(output);


	if (DEBUG)
	{
		cout << Moving2FixAM.size() << endl;
		for (int aa = 0; aa < Moving2FixAM.size(); aa++)
		{
			cout << Moving2FixAM[aa] << "   ";
		}
	}

	return;
}


std::vector<cv::Point2f> FishImageProcess::ZBB2FishTransform(cv::Rect roi)
{

	std::vector<cv::Point2f> regionInFish;


	FishReg.getRegionFromUser(roi);
	//��rotation/crop/affine����������л�ȡ����
	FishReg.getRotationMatrix(-rotationAngleX, rotationAngleY);
	FishReg.getCropPoint(cropPoint);
	FishReg.getFix2MovingAffineMatrix(Moving2FixAM);

	////cout << endl << "1111" << endl;
	////����ת��
	regionInFish = FishReg.ZBB2FishTransform();


	//std::cout << "getRotationMatrix rotationAngleX: " << -rotationAngleX << std::endl;

	
	FishReg.clear();

	return regionInFish;
}




void FishImageProcess::clear()
{
	
	return;
}

void FishImageProcess::freeMemory()
{
	cout << "free cuda memory..." << endl;

	cudaFree(PSF_1_gpu);
	cudaFree(PSF_1_gpu_Complex);
	cudaFree(OTF);
	cudaFree(ImgEst);
	cudaFree(Ratio);
	cudaFree(gpuObjRecon);
	cudaFree(gpuObjRecROI);
	cudaFree(Img_gpu);
	cudaFree(ImgExp);
	cudaFree(gpuObjRecon_Complex);
	cudaFree(float_temp);
	cudaFree(Ratio_Complex);
	cudaFree(fftRatio);
	cudaFree(gpuObjRecon_crop);
	cudaFree(input_image_gpu);
	cudaFree(output_image_gpu);
	cudaFree(imageRotated2D_XY_GPU);
	cudaFree(imageRotated2D_XY_BW_GPU);


	cout << "done" << endl;
	
	cout << "free cpu memory..." << endl;

	delete[] cpuObjRecon;
	delete[] cpuObjRecon_crop;
	delete[] imageRotated2D_XY_BW_CPU;

	cout << "done" << endl;
	return;
}

FishImageProcess::FishImageProcess(const std::string& model_path) :device(torch::kCUDA)
{
	// is CUDA avaliabel??
	//torch::DeviceType device_type;
	if (torch::cuda::is_available())
	{
		device = torch::kCUDA;
		std::cout << "cuda available" << std::endl;
	}
	else
	{
		device = torch::kCPU;
		std::cout << "cuda not avaliable" << std::endl;
	}


	torch::Tensor tensor1 = torch::eye(3); // (A) tensor-cpu
	torch::Tensor tensor2 = torch::eye(3, device); // (B) tensor-cuda
	std::cout << tensor1 << std::endl;
	std::cout << tensor2 << std::endl;


	try
	{
		model = torch::jit::load(model_path);
	}
	catch (const c10::Error& e)
	{
		std::cerr << "Error loading the model!\n";
		std::exit(EXIT_FAILURE);
	}

	//device = torch::kCUDA;
	model.eval();
	model.to(device);
	std::cout << "load model success" << std::endl;
}
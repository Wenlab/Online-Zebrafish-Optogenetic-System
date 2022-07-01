#include"imgProcess.h"
#include"kexinLibs.h"
//#include"kexinLibs.cpp"
#include"initANDcheck.h"
//#include"initANDcheck.cu"
//#include"header.cuh"
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

void FishImageProcess::readImageFromFile(std::string filename)
{
	cout << "start read image from file" << endl;
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

	cout << "read image file done" << endl;

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
	FILE * template_roXY_fid = fopen(filenameXY.data(), "rb");
	if (template_roXY_fid == NULL)
	{
		cout << filenameXY << " open failed!" << endl;
		system("pause");
		return;
	}
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
		{ int(imgSizeAfterCrop_Z), int(imgSizeAfterCrop_Y), int(imgSizeAfterCrop_X) }).toType(torch::kFloat32);
	fixtensor = normalizeTensor(fixtensor);
	cout << "read fix image and convert to normalize tensor" << endl;
	torch::Device device(torch::kCUDA);
	fixtensor.to(device);
	cout << "copy fix tensor to CUDA" << endl;

	//���紦���һ��ͼ���������ǰ��һ��
	model.forward({ fixtensor.to(device),fixtensor.to(device) }).toTensor();

	return;
}

//void FishImageProcess::readModelFromFile(std::string filename)
//{
//
//}

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


	//crop
	cpuObjRotation_crop = new float[200 * 200 * 50];
	check1(cudaMalloc((void**)&ObjCropRed_gpu, sizeof(float)*imgSizeAfterCrop_X*imgSizeAfterCrop_Y*imgSizeAfterCrop_Z),
		"ObjReconRed_gpu cudaMalloc Error", __FILE__, __LINE__);


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
	cufftResult res = cufftPlan2d(&plan, PSF_size_1, PSF_size_2, CUFFT_C2C);



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
		/**********************************************************************************************************/
		/*----������ȷ��Tmp����matlab����47424472��C������47424477.675621979�����ǳ��ǳ�С��Ӧ�ÿ��Ժ���
		�ڶ���matlab��51785136��C������51785130.147748277�����Ҳ�ǳ�С�����Ժ���----*/
		/**********************************************************************************************************/

		////8��Ratio(1:end,1:end)=ImgExp(1:end,1:end)./(ImgEst(1:end,1:end)+Tmp/SNR)����ת�ɸ��������鲿Ϊ��;
		Ratio_Complex_ge << <blockNum_12, threadNum_12 >> > (ImgExp, ImgEst, Tmp, SNR, Ratio_Complex, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Ratio_Complex_ge Error");

		/*******************************************************************************************/
		/*----������ȷ������ĺͣ�matlab��0.3017935��C������0.301793���ڶ�����ȷ-------------------*/
		/*******************************************************************************************/

		////9��fft2(Ratio)
		res = cufftExecC2C(plan, Ratio_Complex, Ratio_Complex, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS)
		{
			cout << "Ratio_Complex cufftExecC2C error:" << res << endl;
			system("pause");
			return;
		}

		/*******************************************************************************************/
		/*----������ȷ������ĺͺ�matlab��һ����������С�������λ��̫���ˣ�����ֵ�ĺ���һ����-------*/
		/*******************************************************************************************/

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


		//float* test = new float[512 * 512 * 50];
		//check(cudaMemcpy(test, gpuObjRecon, sizeof(float) * 512 * 512 * 50, cudaMemcpyDeviceToHost), "gpuObjRecon_crop cudaMemcpy Error");
		//saveAndCheckImage(test, 512, 512, 50, int2string(2, i) + "gpuObjRecon.tif");

		//cout << "��ɵ�" << i << "��ѭ��" << endl << endl << endl;
	}



	cout << "�ع����" << endl;
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
	cout << "line_start: " << line_start <<endl;
	cout << "line_end: " << line_end << endl;
	cout << "line_total: " << line_total << endl;
	cout << "col_start: " << col_start << endl;
	cout << "col_end: " << col_end << endl;
	cout << "col_total: " << col_total << endl;


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


	//cropReconImage_kernel << <blockNum_123, threadNum_123 >> > (gpuObjRecon, gpuObjRecon_crop);
	cout << "crop�ع��������ݲ�copy��GPU" << endl;
	return;
}


void FishImageProcess::matchingANDrotationXY()
{
	/*   XYƽ���ģ��ƥ�����ת   */
	cout << "start XY 2D template matching..." << endl;
	dim3 block_1(32, 32, 1);
	dim3 grid_1((200 + block_1.x - 1) / block_1.x, (200 + block_1.y - 1) / block_1.y, 1);
	kernel_1 << <grid_1, block_1 >> > (gpuObjRecon_crop, 200, 200, image2D_XY_gpu);   
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_1 Error");

	thrust::device_ptr<float> dev_ptr(image2D_XY_gpu);
	double image2D_XY_mean = thrust::reduce(dev_ptr, dev_ptr + size_t(200 * 200), (float)0, thrust::plus<float>()) / (200 * 200);
	cout << "image2D_XY_mean: " << image2D_XY_mean << endl;

	int threadNum_2 = 256;
	int blockNum_2 = (200 * 200 - 1) / threadNum_2 + 1;
	kernel_2 << <blockNum_2, threadNum_2 >> > (image2D_XY_gpu, 200 * 200, image2D_XY_mean, img2DBW_XY_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_2 Error");

	int threadNum_3 = 256;
	int blockNum_3 = (rotationAngleXY_size - 1) / threadNum_2 + 1;
	kernel_3 << <blockNum_3, threadNum_3 >> > (template_roXY_gpu, img2DBW_XY_gpu, rotationAngleXY_size, err_XY_gpu);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_3 Error");


	//��err_XY_gpu����Сֵ
	double *err_XY = new double[rotationAngleXY_size];
	check(cudaMemcpy(err_XY, err_XY_gpu, sizeof(double)*rotationAngleXY_size, cudaMemcpyDeviceToHost), "err_XY cudaMemcpy Error");
	double err_XY_min = DBL_MAX;
	int idx;  //�ҵ���Сֵ��Ӧ������
	for (int i = 0; i < rotationAngleXY_size; i++)
	{
		//cout << i << "   " << err_XY[i] << endl;
		if (err_XY[i] < err_XY_min)
		{
			err_XY_min = err_XY[i];
			idx = i;
		}
	}
	//cout << "err_XY_min: " << err_XY_min << endl;
	//cout << "rotation XY idx: " << idx << endl;
	//��һ����ת

	ObjRecon_imrotate3_gpu(gpuObjRecon_crop, -rotationAngleXY[idx], imageRotated3D_gpu);
	cout << "XY 2D templaet matching and rotation done" << endl;

	return;
}

void FishImageProcess::ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu)
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


//void FishImageProcess::matchingANDrotationYZ()
//{
//	/*  YZƽ���ģ��ƥ�����ת  */
////�� y-z���ͶӰ,����imageRotated3Dһ���������з�������ֵ
//	float *image2D_YZ_gpu;
//	check1(cudaMalloc((void**)&image2D_YZ_gpu, sizeof(float) * 200 * 50), "image2D_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);//200��*50�а���imageRotated3D�����������У���matlab�а�������������
//	dim3 block_4(32, 32, 1);
//	dim3 grid_4((50 + block_4.x - 1) / block_4.x, (200 + block_4.y - 1) / block_4.y, 1);
//	kernel_4 << <grid_4, block_4 >> > (imageRotated3D_gpu, image2D_YZ_gpu);
//	cudaDeviceSynchronize();
//	checkGPUStatus(cudaGetLastError(), "kernel_4 Error");
//	//image2D_YZ_gpu��͡���ֵ
//	thrust::device_ptr<float> dev_ptr1(image2D_YZ_gpu);
//	double image2D_YZ_mean = thrust::reduce(dev_ptr1, dev_ptr1 + size_t(200 * 50), (float)0, thrust::plus<float>()) / (200 * 50) + 14;
//	//��ֵ�� y-z�棬����mean��ȡ1�� С�ڵ���mean��ȡ0
//	float *img2DBW_YZ_gpu;
//	check1(cudaMalloc((void**)&img2DBW_YZ_gpu, sizeof(float) * 200 * 50), "img2DBW_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);
//	int threadNum_5 = 256;
//	int blockNum_5 = (200 * 50 - 1) / threadNum_5 + 1;
//	kernel_5 << <blockNum_5, threadNum_5 >> > (image2D_YZ_gpu, image2D_YZ_mean, img2DBW_YZ_gpu);
//	cudaDeviceSynchronize();
//	checkGPUStatus(cudaGetLastError(), "kernel_5 Error");
//
//
//	//��ÿ���Ƕȵ���� ��ʼ��
//	float *template_roYZ_gpu;
//	check1(cudaMalloc((void**)&template_roYZ_gpu, sizeof(float) * template_roYZ_size), "template_roYZ_gpu cudaMalloc Error", __FILE__, __LINE__);
//	check(cudaMemcpy(template_roYZ_gpu, template_roYZ, sizeof(float)*template_roYZ_size, cudaMemcpyHostToDevice), "template_roYZ_gpu cudaMemcpy Error");
//	double *err_YZ_gpu;
//	check1(cudaMalloc((void**)&err_YZ_gpu, sizeof(double) * rotationAngleYZ_size), "err_YZ_gpu cudaMalloc Error", __FILE__, __LINE__);
//	int threadNum_6 = 256;
//	int blockNum_6 = (rotationAngleYZ_size - 1) / threadNum_6 + 1;
//	kernel_6 << <blockNum_5, threadNum_5 >> > (template_roYZ_gpu, img2DBW_YZ_gpu, rotationAngleYZ_size, err_YZ_gpu);
//	cudaDeviceSynchronize();
//	checkGPUStatus(cudaGetLastError(), "kernel_6 Error");
//	//��err_YZ_gpu����Сֵ����Сֵ������
//	double *err_YZ = new double[rotationAngleYZ_size];
//	check(cudaMemcpy(err_YZ, err_YZ_gpu, sizeof(double)*rotationAngleYZ_size, cudaMemcpyDeviceToHost), "err_YZ cudaMemcpy Error");
//	double err_YZ_min = DBL_MAX;
//	for (int i = 0; i < rotationAngleYZ_size; i++)
//	{
//		if (err_YZ[i] < err_YZ_min)
//			err_YZ_min = err_YZ[i];
//	}
//	int idx2;
//	for (int i = 0; i < rotationAngleYZ_size; i++)
//	{
//		if (err_YZ[i] == err_YZ_min)
//		{
//			idx2 = i;
//			break;
//		}
//	}
//	//imageRotated3D��ת����X������תrotationAngleYZ(idx2)��
//	//�Ȱ�imageRotated3D_gpu��ά�ȱ任һ�£��б�ɲ��Σ����α���У��б�ɷ��ţ�(200 * 200 * 50)���(200�� * 50�� * 200)����
//	float *imageRotated3D_gpu_1;
//	check1(cudaMalloc((void**)&imageRotated3D_gpu_1, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_1 cudaMalloc Error", __FILE__, __LINE__);
//	dim3 block_7(8, 8, 8);
//	dim3 grid_7((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
//	kernel_7 << <grid_7, block_7 >> > (imageRotated3D_gpu, imageRotated3D_gpu_1);
//	cudaDeviceSynchronize();
//	checkGPUStatus(cudaGetLastError(), "kernel_7 Error");
//	//�ڶ�����ת
//	float *imageRotated3D_gpu_2;
//	check1(cudaMalloc((void**)&imageRotated3D_gpu_2, sizeof(float)*ObjRecon_size), "imageRotated3D_gpu_2 cudaMalloc Error", __FILE__, __LINE__);
//	ObjRecon_imrotate3_X_gpu(imageRotated3D_gpu_1, rotationAngleYZ[idx2], imageRotated3D_gpu_2);
//	//�ٰ�ά�ȱ任��ԭ����
//	dim3 block_8(8, 8, 8);
//	dim3 grid_8((200 + block_7.x - 1) / block_7.x, (200 + block_7.y - 1) / block_7.y, (50 + block_7.z - 1) / block_7.z);
//	kernel_8 << <grid_8, block_8 >> > (imageRotated3D_gpu_2, imageRotated3D_gpu);
//	cudaDeviceSynchronize();
//	checkGPUStatus(cudaGetLastError(), "kernel_8 Error");
//
//
//	return;
//}




void FishImageProcess::cropRotatedImage()
{
	cout << "start crop rotation image..." << endl;

	//	//����imageRotated3D_gpu�ľ�ֵ
	thrust::device_ptr<float> dev_ptr2(imageRotated3D_gpu);
	double imageRotated3D_x_mean = thrust::reduce(dev_ptr2, dev_ptr2 + size_t(ObjRecon_size), (float)0, thrust::plus<float>()) / (ObjRecon_size)+4;

	check(cudaMemcpy(cpuObjRotation_crop, imageRotated3D_gpu, sizeof(float)*ObjRecon_size, cudaMemcpyDeviceToHost), "ObjRecon cudaMemcpy Error");

	//crop
	int *idx_2 = new int[ObjRecon_size]();//imageRotated3D_x���ھ�ֵ������
	int idx_2_size = 0;
	for (int i = 0; i < ObjRecon_size; i++)
	{
		if (cpuObjRotation_crop[i] > imageRotated3D_x_mean)
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
	cout <<"CentroID: "<< CentroID[0] << "   " << CentroID[1] << "  " << CentroID[2] << endl;
	//CentroID������matlab����[89,91,24]���Ҽ������[86,91,24],x���3����npp��ת��matlab�Ľ���������ɵģ����Ҳ��������������

	// ��������������Χ����������������matlab������Ҫ��ȥ1
	// �з�Χ����CentroID(0)-61��CentroID(0)+33�� ���з�Χ����CentroID(2)-38��CentroID(2)+37�������еĲ���
	//int XObj = CentroID[0] + 33 - (CentroID[0] - 61) + 1;//��
	//int	YObj = CentroID[2] + 37 - (CentroID[2] - 38) + 1;//��
	//int	ZObj = 50;//����

	if (CentroID[0] < 61 || CentroID[1] < 38 || CentroID[0]>167 || CentroID[1]>163)
	{
		cout << "centroID error!!!" << endl;
		return;
	}


	dim3 block_10(8, 8, 8);
	dim3 grid_10((imgSizeAfterCrop_X + block_10.x - 1) / block_10.x, (imgSizeAfterCrop_Y + block_10.y - 1) / block_10.y, (imgSizeAfterCrop_Z + block_10.z - 1) / block_10.z);
	//__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2)
	kernel_10 << <grid_10, block_10 >> > (imageRotated3D_gpu, ObjCropRed_gpu, imgSizeAfterCrop_X, imgSizeAfterCrop_Y, imgSizeAfterCrop_Z, CentroID[0], CentroID[1]);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "kernel_10 Error");
	cout << "crop���" << endl;

	return;
}





void FishImageProcess::libtorchModelProcess()
{
	//libtorch
	//convert image to tensor
	torch::Tensor movingtensor;
	movingtensor = torch::from_blob(ObjCropRed_gpu,
		{ int(imgSizeAfterCrop_Z), int(imgSizeAfterCrop_Y), int(imgSizeAfterCrop_X) }, torch::kCUDA).toType(torch::kFloat32);
	movingtensor = normalizeTensor(movingtensor);
	cout << movingtensor.sizes() << endl;
	cout << fixtensor.sizes() << endl;
	cout << "1111" << endl;
	auto output = model.forward({ movingtensor.to(device),fixtensor.to(device) }).toTensor();
	//auto output = model.forward({ movingtensor,fixtensor }).toTensor();
	cout << "2222" << endl;
	std::vector<float> Moving2FixAM = rescaleAffineMatrix(output);


	if (1)
	{
		cout << Moving2FixAM.size() << endl;
		for (int aa = 0; aa < Moving2FixAM.size(); aa++)
		{
			cout << Moving2FixAM[aa] << "   ";
		}
	}
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

	cout << "done" << endl;
	
	cout << "free cpu memory..." << endl;

	free(cpuObjRecon);
	free(cpuObjRecon_crop);

	cout << "done" << endl;
	return;
}

FishImageProcess::FishImageProcess(const std::string& model_path) :device(torch::kCPU)
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
	try
	{
		model = torch::jit::load(model_path);
	}
	catch (const c10::Error& e)
	{
		std::cerr << "Error loading the model!\n";
		std::exit(EXIT_FAILURE);
	}
	model.eval();
	model.to(device);
	std::cout << "load model success" << std::endl;
}
#pragma once
#include <string>
#include <stdlib.h>
#include <vector>
#include <cufft.h>

//libtorch
#include <torch/torch.h>
#include <torch/script.h>

#include <complex>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>

//����ת��
#include"zbb2FishImg.h"

class FishImageProcess
{
public:
	FishImageProcess(const std::string& model_path);
	//~FishImageProcess();

	float *PSF_1;
	unsigned short *Img; //���ļ���ȡ��ͼ��
	float *template_roXY;  //XYƽ���ģ�壬��360��
	float *template_roYZ;  //YZƽ���ģ�壬��31��
	double *rotationAngleXY;   //XYƽ�����ת�Ƕ�
	double *rotationAngleYZ; //YZƽ�����ת�Ƕ�

	//libtorch
	torch::DeviceType device_type;
	torch::Device device;
	torch::jit::script::Module model;
	torch::Tensor fixtensor;  //fix image pytorch tensor
	torch::Tensor movingtensor;  //moving image pytorch tensor

	cufftHandle fftplanfwd;//�������

	//��ʼ���ڴ�
	//�ع�
	float *PSF_1_gpu;
	cufftComplex *PSF_1_gpu_Complex;
	cufftComplex *OTF;
	float *ImgEst;
	float *Ratio;
	float *gpuObjRecon;
	float *gpuObjRecROI;
	unsigned short *Img_gpu;
	float *ImgExp;
	cufftComplex *gpuObjRecon_Complex;
	float *float_temp;
	cufftComplex *Ratio_Complex;
	cufftComplex *fftRatio;

	float *cpuObjRecon;
	float *cpuObjRecon_crop;   //CPU�ϴ洢crop������ObjRecon
	float *gpuObjRecon_crop;   //GPU�ϴ洢crop���ObjRecon, �ع����

	//��ת
	float *image2D_XY_gpu;
	float *img2DBW_XY_gpu;
	float *template_roXY_gpu;
	double *err_XY_gpu;
	float *imageRotated3D_gpu;    //��ת���

	//crop
	float *cpuObjRotation_crop;
	float *ObjCropRed_gpu;    //crop���

	//����ת��
	zbb2FishImg coordTrans;


	//�ع�ǰ��ͼ���С
	int imgSizeBeforeRecon_X = 512;
	int imgSizeBeforeRecon_Y = 512;
	int imgSizeBeforeRecon_Z = 1;
	//�ع����ͼ���С
	int imgSizeAfterRecon_X = 200;
	int imgSizeAfterRecon_Y = 200;
	int imgSizeAfterRecon_Z = 50;
	//��ת��crop���ͼ���С
	int imgSizeAfterCrop_X = 76;
	int imgSizeAfterCrop_Y = 95;
	int imgSizeAfterCrop_Z = 50;

	//ģ���С
	int template_roXY_size = 200 * 200 * 360;
	int rotationAngleXY_size = 360;

	void readPSFfromFile(std::string filename);
	void readImageFromFile(std::string filename);   
	void readImageFromCamera(std::string filename);
	void readTemplateFromFile(std::string filenameTemXY, std::string filenameTemYZ);
	void readRotationAngleFromFile(std::string filenameAngleXY, std::string filenameAngleYZ);
	void readFixImageFromFile(std::string filename);

	//void readModelFromFile(std::string filename);  //�ڹ��캯������ȡ
	

	void prepareGPUMemory();
	void processPSF();   //PSF Ԥ����ֻ��Ҫ����һ�� 
	void reconImage();   //��ÿ��ͼ���ع�������ͼ����Ҫѭ������
	void cropReconImage();   //�ع�������ͼ����512*512*50�� crop��200*200*50   ��һ������ʡ�ԣ�����
	void matchingANDrotationXY();  //XYƽ���ģ��ƥ�����ת
	void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu);
	//void matchingANDrotationYZ();  //YZƽ���ģ��ƥ�����ת
	void cropRotatedImage();   //200*200*50  crop��95*76*50
	void libtorchModelProcess();


	void clear();   //�������
	void freeMemory();  //�����ڴ�

private:

};



//FishImageProcess::~FishImageProcess()
//{
//}


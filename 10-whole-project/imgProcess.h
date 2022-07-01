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

//坐标转换
#include"zbb2FishImg.h"

class FishImageProcess
{
public:
	FishImageProcess(const std::string& model_path);
	//~FishImageProcess();

	float *PSF_1;
	unsigned short *Img; //从文件读取的图像
	float *template_roXY;  //XY平面的模板，共360个
	float *template_roYZ;  //YZ平面的模板，共31个
	double *rotationAngleXY;   //XY平面的旋转角度
	double *rotationAngleYZ; //YZ平面的旋转角度

	//libtorch
	torch::DeviceType device_type;
	torch::Device device;
	torch::jit::script::Module model;
	torch::Tensor fixtensor;  //fix image pytorch tensor
	torch::Tensor movingtensor;  //moving image pytorch tensor

	cufftHandle fftplanfwd;//创建句柄

	//初始化内存
	//重构
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
	float *cpuObjRecon_crop;   //CPU上存储crop出来的ObjRecon
	float *gpuObjRecon_crop;   //GPU上存储crop后的ObjRecon, 重构结果

	//旋转
	float *image2D_XY_gpu;
	float *img2DBW_XY_gpu;
	float *template_roXY_gpu;
	double *err_XY_gpu;
	float *imageRotated3D_gpu;    //旋转结果

	//crop
	float *cpuObjRotation_crop;
	float *ObjCropRed_gpu;    //crop结果

	//坐标转换
	zbb2FishImg coordTrans;


	//重构前的图像大小
	int imgSizeBeforeRecon_X = 512;
	int imgSizeBeforeRecon_Y = 512;
	int imgSizeBeforeRecon_Z = 1;
	//重构后的图像大小
	int imgSizeAfterRecon_X = 200;
	int imgSizeAfterRecon_Y = 200;
	int imgSizeAfterRecon_Z = 50;
	//旋转和crop后的图像大小
	int imgSizeAfterCrop_X = 76;
	int imgSizeAfterCrop_Y = 95;
	int imgSizeAfterCrop_Z = 50;

	//模板大小
	int template_roXY_size = 200 * 200 * 360;
	int rotationAngleXY_size = 360;

	void readPSFfromFile(std::string filename);
	void readImageFromFile(std::string filename);   
	void readImageFromCamera(std::string filename);
	void readTemplateFromFile(std::string filenameTemXY, std::string filenameTemYZ);
	void readRotationAngleFromFile(std::string filenameAngleXY, std::string filenameAngleYZ);
	void readFixImageFromFile(std::string filename);

	//void readModelFromFile(std::string filename);  //在构造函数处读取
	

	void prepareGPUMemory();
	void processPSF();   //PSF 预处理，只需要运行一次 
	void reconImage();   //对每张图像重构，多张图像需要循环运行
	void cropReconImage();   //重构出来的图像是512*512*50， crop到200*200*50   这一步可以省略？？？
	void matchingANDrotationXY();  //XY平面的模板匹配和旋转
	void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu);
	//void matchingANDrotationYZ();  //YZ平面的模板匹配和旋转
	void cropRotatedImage();   //200*200*50  crop成95*76*50
	void libtorchModelProcess();


	void clear();   //清除变量
	void freeMemory();  //回收内存

private:

};



//FishImageProcess::~FishImageProcess()
//{
//}


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

#include"zbb2FishImg.h"

#define DEBUG 0

class FishImageProcess
{
public:
	FishImageProcess(const std::string& model_path);

	float *PSF_1;
	unsigned short *Img; //Image read from file
	float *template_roXY;  
	float *template_roYZ;  
	double *rotationAngleXY;   //Rotation angle of XY plane
	double *rotationAngleYZ; //Rotation angle of YZ plane

	//libtorch
	torch::DeviceType device_type;
	torch::Device device;
	torch::jit::script::Module model;
	torch::Tensor fixtensor;  //fix image pytorch tensor
	torch::Tensor movingtensor;  //moving image pytorch tensor

	cufftHandle fftplanfwd;

	//Initialize memory
	//Reconstruction
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
	float *cpuObjRecon_crop;  
	float *gpuObjRecon_crop;  

	//Rotation
	float *input_image_gpu;  //before rotation
	float *output_image_gpu;  //after rotation
	float *image2D_XY_gpu;
	float *img2DBW_XY_gpu;
	float *template_roXY_gpu;
	double *err_XY_gpu;
	float *imageRotated3D_gpu;    //rotation result

	//crop
	float *cpuObjRotation_crop;
	float *ObjCropRed_gpu;    //crop result
	float* imageRotated2D_XY_GPU;
	float* imageRotated2D_XY_BW_GPU;
	float* imageRotated2D_XY_BW_CPU;


	//Coordinate conversion
	int rotationAngleX = 0;
	int rotationAngleY = 0;
	cv::Point3d cropPoint;
	std::vector<float> Moving2FixAM;
	zbb2FishImg FishReg;


	//Image size before reconstruction
	int imgSizeBeforeRecon_X = 512;
	int imgSizeBeforeRecon_Y = 512;
	int imgSizeBeforeRecon_Z = 1;
	//Size of reconstructed image
	int imgSizeAfterRecon_X = 200;
	int imgSizeAfterRecon_Y = 200;
	int imgSizeAfterRecon_Z = 50;
	//Image size after rotate and crop
	int imgSizeAfterCrop_X = 76;
	int imgSizeAfterCrop_Y = 95;
	int imgSizeAfterCrop_Z = 50;

	//Template size
	int template_roXY_size = 200 * 200 * 360;
	int rotationAngleXY_size = 360;


	void initialize();


	void readPSFfromFile(std::string filename);
	void loadImage(unsigned short * imgbuffer);
	void readImageFromFile(std::string filename);
	void readImageFromCamera(std::string filename);
	void readTemplateFromFile(std::string filenameTemXY, std::string filenameTemYZ);
	void readRotationAngleFromFile(std::string filenameAngleXY, std::string filenameAngleYZ);
	void readFixImageFromFile(std::string filename);
	void initializeFishReg(std::string filename);
	

	void prepareGPUMemory();
	void processPSF();   //PSF pre-processing, run only once
	void reconImage();   //Reconstruction for each image, multiple images need to be run in a loop
	void cropReconImage();   //The reconstructed image is 512*512*50, crop to 200*200*50
	void matchingANDrotationXY();  //Template matching and rotation in XY plane
	void ObjRecon_imrotate3_gpu(float *ObjRecon_gpu, double nAngle, float *imageRotated3D_gpu);
	//void matchingANDrotationYZ();  //Template matching and rotation of YZ plane
	void cropRotatedImage(int xbias, int ybias);   //200*200*50  crop to 95*76*50
	void libtorchModelProcess();
	std::vector<cv::Point2f> ZBB2FishTransform(cv::Rect roi);

	void clear(); 
	void freeMemory(); 

private:

};



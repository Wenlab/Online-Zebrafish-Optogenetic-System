#include <torch/torch.h>
#include <torch/script.h>

#include <direct.h> 

#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkWindowedSincInterpolateImageFunction.h"

#include "itkImage.h"
#include "itkRandomImageSource.h"

#include "itkImageFileReader.h"

#include "itkImageFileWriter.h"
#include "itkTIFFImageIOFactory.h"
#include "itkNIFTIImageIOFactory.h"
#include "itkNiftiImageIO.h"

#include "itkAffineTransform.h"
#include "itkResampleImageFilter.h"

#include "itkPasteImageFilter.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkMinimumMaximumImageCalculator.h"
#include "itkDivideImageFilter.h"

#include <string>
#include <iostream>
#include<vector>
#include<io.h>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#define _USE_MATH_DEFINES
#include <math.h>

#include"Timer.h"

constexpr unsigned int Dimension = 3;
using InputPixelType = float;
using ImageType = itk::Image<InputPixelType, Dimension>;
using ReaderType = itk::ImageFileReader<ImageType>;

itk::Image<float, 3>::Pointer warpImage(itk::Image<float, 3>::Pointer movingImg, std::vector<float> affineMatrix);
std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix);
torch::Tensor convertITKImg2Tensor(itk::Image<float, 3>* img);
void getFileNames(std::string path, std::vector<std::string>& files);
std::vector<float> reverseAffineMatrix(std::vector<float> affineMatrix);
std::string int2string(int n, int i);

int main()
{
	std::string path = "F:/ITK/initialization/matlab/standardZBBtest/StandardDatas/01/";
	//get all image file name
	std::vector<std::string> fixfilenames;
	getFileNames(path+"fix/", fixfilenames);
	std::vector<std::string> movingfilenames;
	getFileNames(path+"moving/", movingfilenames);
	if (fixfilenames.size() == movingfilenames.size())
	{
		std::cout << "fix file num : " << fixfilenames.size() << std::endl;
		std::cout << "moving file num : " << movingfilenames.size() << std::endl;
		std::cout << "okk, fix image num equal to moving image num" << std::endl;
	}
	else
	{
		std::cout << "fix file num : " << fixfilenames.size() << std::endl;
		std::cout << "moving file num : " << movingfilenames.size() << std::endl;
		std::cout << "error!! fix img num not equal to moving img" << std::endl;
	}

	// is CUDA avaliabel??
	torch::DeviceType device_type;
	if (torch::cuda::is_available())
	{
		device_type = torch::kCUDA;
		std::cout << "cuda available" << std::endl;
	}
	else
	{
		device_type = torch::kCPU;
		std::cout << "cuda not avaliable" << std::endl;
	}

	torch::Device device(device_type);

	// load model
	auto model = torch::jit::load("affineNetScript_TM_cpu.pt");
	model.eval();
	model.to(device);
	std::cout << "load model success" << std::endl;


	//read image with ITK
	//The result of affine transformation of movingImage
	std::string resultpath = path + "result1/";
	int ret = _access(resultpath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(resultpath.c_str());
	}

	// After finding the inverse of affineMatrix, inverse affine to fixImg
	// Check if the inverse affine results are correct
	std::string doubleAffinePath = path+"result2/";
	ret = _access(doubleAffinePath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(doubleAffinePath.c_str());
	}
	std::string finallyPath = path+"result3/";
	ret = _access(finallyPath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(finallyPath.c_str());
	}


	std::string filename = path+"fix2movingParam.txt";
	//write out
	std::cout << "write affine params to txt...." << std::endl;
	std::ofstream fOut(filename);
	if (!fOut)
	{
		std::cout << "Open output file faild." << std::endl;
	}




	for (int i = 1; i < fixfilenames.size(); i++)
	//for (int i = 0; i < 5; i++)
	{

		ReaderType::Pointer reader1 = ReaderType::New();
		reader1->SetFileName(fixfilenames[i]);
		itk::NiftiImageIOFactory::RegisterOneFactory();
		reader1->Update();
		ImageType* fix = reader1->GetOutput();


		ReaderType::Pointer reader2 = ReaderType::New();
		reader2->SetFileName(movingfilenames[i]);
		reader2->Update();
		ImageType* moving = reader2->GetOutput();

		//convert image to tensor
		torch::Tensor fixtensor = convertITKImg2Tensor(fix);
		torch::Tensor movingtensor = convertITKImg2Tensor(moving);
		//std::cout << "tensor size: " << fixtensor.sizes() << std::endl;


		auto output = model.forward({ movingtensor.to(device),fixtensor.to(device) }).toTensor();




		std::vector<float> am = rescaleAffineMatrix(output);

		for (int k = 0; k < am.size(); k++)
		{
			fOut << am[k] << "   ";
		}
		fOut << std::endl;

		ImageType::Pointer afterAffineImg = warpImage(moving, am);
		//std::cout << "warp result size" << afterAffineImg->GetLargestPossibleRegion().GetSize() << std::endl;
		try
		{
			itk::WriteImage(afterAffineImg, resultpath + int2string(4, i) + ".nii");
			std::cout << i << std::endl;
		}
		catch (itk::ExceptionObject & error)
		{
			std::cerr << "Error: " << error << std::endl;
			return EXIT_FAILURE;
		}

		//After finding the inverse of affineMatrix, inverse affine to fixImg
		std::vector<float> am_inverse = reverseAffineMatrix(am);
		//warp image
		ImageType::Pointer InverseAffineFixImg = warpImage(fix, am_inverse);
		try
		{
			itk::WriteImage(InverseAffineFixImg, doubleAffinePath + int2string(4, i) + ".nii");
			std::cout << i << std::endl;
		}
		catch (itk::ExceptionObject & error)
		{
			std::cerr << "Error: " << error << std::endl;
			return EXIT_FAILURE;
		}

	}


	fOut.close();
	std::cout << "done" << std::endl;
	return 0;
}

torch::Tensor convertITKImg2Tensor(itk::Image<float, 3>* img)
{
	using InputImageType = itk::Image<float, 3>;
	InputImageType::SizeType size = img->GetLargestPossibleRegion().GetSize();
	//std::cout << "orig image size: "<<size << std::endl;

	//convert itk image to buffer
	float * buffer = new float[size[0] * size[1] * size[2]];
	using IteratorType = itk::ImageRegionConstIterator<InputImageType>;
	IteratorType it(img, img->GetLargestPossibleRegion());
	size_t buffer_index(0);
	it.GoToBegin();
	while (!it.IsAtEnd())
	{
		buffer[buffer_index] = it.Get();
		++it;
		++buffer_index;
	}

	

	// convert buffer to tensor
	//itk  77*96*50
	//tensor  50*96*77
	//The size here can not follow the imageSize of itk
	//In python, sitk to numpy will swap the first and third dimensions
	torch::Tensor tensor = torch::from_blob(buffer,
		{ int(size[2]), int(size[1]), int(size[0]) }).toType(torch::kFloat32);  

	auto max_result = torch::max(tensor);
	auto min_result = torch::min(tensor);


	//normalize
	tensor = (tensor-min_result)/(max_result-min_result);

	tensor = tensor.unsqueeze(0);

	return tensor;
}

std::vector<float> reverseAffineMatrix(std::vector<float> am)
{
	
	Eigen::Matrix3f am_eigen;  //Does not contain three panning parameters
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_eigen(b, a) = am[a * 3 + b];
		}
	}
	Eigen::Matrix3f am_eigen_inverse = am_eigen.inverse();

	Eigen::Vector3f am_trans;
	am_trans << am[9], am[10], am[11];
	Eigen::Vector3f am_trans_inverse = am_trans.transpose() * (-am_eigen_inverse);


	std::vector<float> am_inverse(12);
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_inverse[a * 3 + b] = am_eigen_inverse(b, a);
		}
	}

	am_inverse[9] = am_trans_inverse(0);
	am_inverse[10] = am_trans_inverse(1);
	am_inverse[11] = am_trans_inverse(2);

	return am_inverse;
}

itk::Image<float, 3>::Pointer warpImage(itk::Image<float, 3>::Pointer moving, std::vector<float> am)
{
	//std::cout << "moving image size" << moving->GetLargestPossibleRegion().GetSize() << std::endl;
	using ResampleImageFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
	ResampleImageFilterType::Pointer resample = ResampleImageFilterType::New();
	resample->SetInput(moving);
	resample->SetReferenceImage(moving);
	resample->UseReferenceImageOn();
	resample->SetSize(moving->GetLargestPossibleRegion().GetSize());
	resample->SetDefaultPixelValue(0);

	using InterpolatorType = itk::WindowedSincInterpolateImageFunction<ImageType, 3>;
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	resample->SetInterpolator(interpolator);

	using TransformType = itk::AffineTransform < double, Dimension >;
	TransformType::Pointer transform = TransformType::New();
	TransformType::ParametersType parameters(Dimension * Dimension + Dimension);
	for (int i = 0; i < am.size(); i++)
	{
		parameters[i] = am[i];
		//std::cout << am[i] << std::endl;
	}
	transform->SetParameters(parameters);
	resample->SetTransform(transform);
	resample->Update();

	return resample->GetOutput();
}

std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix)
{
	//affineMatrix: 1*12
	affineMatrix=affineMatrix.squeeze(0);
	affineMatrix = affineMatrix / 100.0;

	affineMatrix = affineMatrix.to(torch::kCPU);
	std::vector<float> v(affineMatrix.data_ptr<float>(), affineMatrix.data_ptr<float>() + affineMatrix.numel());

	v[0] = v[0] / 10 + 1;
	v[4] = v[4] / 10 + 1;
	v[8] = v[8] / 10 + 1;
	v[9] = v[9] * 77;
	v[10] = v[10] * 95;
	v[11] = v[11] * 52;

	return v;
}


void getFileNames(std::string path, std::vector<std::string>& files)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFileNames(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


std::string int2string(int n, int i)
{
	char s[BUFSIZ];
	sprintf_s(s, "%d", i);
	int l = strlen(s);

	if (l > n)
	{
		std::cout << "The length of the integer is greater than the length of the string to be formatted£¡";
	}
	else
	{
		std::stringstream M_num;
		for (int i = 0; i < n - l; i++)
			M_num << "0";
		M_num << i;

		return M_num.str();
	}
}
#include <torch/torch.h>
#include <torch/script.h>

#include <direct.h> //_mkdir fun

#include "itkResampleImageFilter.h"
#include "itkAffineTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkWindowedSincInterpolateImageFunction.h"

#include "itkImage.h"
#include "itkRandomImageSource.h"


//包含这个文件时编译会失败???
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
	//get all image file name
	std::vector<std::string> fixfilenames;
	getFileNames("C:/Users/USER/source/repos/Project3/Project3/0824_0924CMTK_AM_TM3_x10_delete/fixImg", fixfilenames);
	std::vector<std::string> movingfilenames;
	getFileNames("C:/Users/USER/source/repos/Project3/Project3/0824_0924CMTK_AM_TM3_x10_delete/movingImg", movingfilenames);
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
	//device_type = torch::kCPU;
	torch::Device device(device_type);

	// load model
	auto model = torch::jit::load("affineNetScript_TM_cpu.pt");
	model.eval();
	model.to(device);
	std::cout << "load model success" << std::endl;


	//read image with ITK
	std::string resultpath = "C:\\Users\\USER\\source\\repos\\Project3\\Project3\\0824_0924CMTK_AM_TM3_x10_delete\\result\\";
	int ret = _access(resultpath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(resultpath.c_str());
	}
	std::string doubleAffinePath = "C:\\Users\\USER\\source\\repos\\Project3\\Project3\\0824_0924CMTK_AM_TM3_x10_delete\\result2\\";
	ret = _access(doubleAffinePath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(doubleAffinePath.c_str());
	}
	std::string finallyPath = "C:\\Users\\USER\\source\\repos\\Project3\\Project3\\0824_0924CMTK_AM_TM3_x10_delete\\result3\\";
	ret = _access(finallyPath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(finallyPath.c_str());
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


		// forward
		//Timer timer;
		//timer.start();
		//for (int j = 0; j < 1000; j++)
		//{
		auto output = model.forward({ movingtensor.to(device),fixtensor.to(device) }).toTensor();
		//}
		//timer.stop();
		//std::cout <<"模型运行1000次所需时间：" <<timer.getElapsedTimeInMilliSec() << std::endl;
		//std::cout << output << std::endl;
		std::vector<float> am = rescaleAffineMatrix(output);

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

		// 求affineMatrix的逆运算后，对fixImg逆仿射
		std::vector<float> am_inverse = reverseAffineMatrix(am);
		//warp image
		ImageType::Pointer InverseAffineFixImg= warpImage(fix, am_inverse);
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

		//还原crop

		//新建一个200*200*50的图像（crop之前的图像大小）
		ImageType::Pointer canvas = ImageType::New();
		ImageType::IndexType origin; //创建itk::Index对象,用来指定图像起点位置
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		ImageType::SizeType size;   //创建itk::Size对象,指定图像各方向大小
		size[0] = 200;
		size[1] = 200;
		size[2] = 50;

		ImageType::RegionType LargeRegion; //创建图像区域，并设置起点和大小
		LargeRegion.SetSize(size);
		LargeRegion.SetIndex(origin);

		canvas->SetRegions(LargeRegion);
		canvas->Allocate();   //注意：这里才真正给image对象分配内存


		//选择一块区域，将图像复制上去
		const auto startx = static_cast<itk::IndexValueType>(55);
		const auto endx = static_cast<itk::IndexValueType>(55+77);

		const auto starty = static_cast<itk::IndexValueType>(64);
		const auto endy = static_cast<itk::IndexValueType>(64+95);

		const auto startz = static_cast<itk::IndexValueType>(0);
		const auto endz = static_cast<itk::IndexValueType>(50);

		ImageType::IndexType start;
		start[0] = startx;
		start[1] = starty;
		start[2] = startz;

		ImageType::IndexType end;
		end[0] = endx;
		end[1] = endy;
		end[2] = endz;

		ImageType::RegionType region;
		region.SetIndex(start);
		region.SetUpperIndex(end);

		using FilterType = itk::PasteImageFilter<ImageType, ImageType>;
		FilterType::Pointer filter = FilterType::New();
		filter->SetSourceImage(InverseAffineFixImg);
		filter->SetSourceRegion(InverseAffineFixImg->GetLargestPossibleRegion());
		filter->SetDestinationImage(canvas);
		filter->SetDestinationIndex(start);

		filter->Update();
		ImageType::Pointer origSizeImage = filter->GetOutput();

		//rotation
		float rotationAngleYZ = 0;
		float rotationAngleXY = 60;
		float yaw = rotationAngleXY * M_PI / 180;
		float pitch = 0 * M_PI / 180;
		float roll = rotationAngleYZ * M_PI / 180;

		Eigen::Matrix3f m;
		m = Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()) *   //绕x轴的旋转，YZ平面的旋转
			Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *  //绕y轴的旋转，XZ平面的旋转
			Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());		//绕z轴的旋转，XY平面的旋转
		std::vector<float> rotationMatrix;
		for (int p = 0; p < m.rows(); p++)
		{
			for (int q = 0; q < m.cols(); q++)
			{
				rotationMatrix.push_back(m(p, q));
			}
		}
		rotationMatrix.push_back(0);   //平移分量为0
		rotationMatrix.push_back(0);
		rotationMatrix.push_back(0);


		//先将图像平移到原点为图像中心
		using ResampleImageFilterType = itk::ResampleImageFilter<ImageType, ImageType>;
		ResampleImageFilterType::Pointer filter = ResampleImageFilterType::New();
		resample->SetInput(moving);
		resample->SetReferenceImage(moving);
		resample->UseReferenceImageOn();
		resample->SetSize(moving->GetLargestPossibleRegion().GetSize());
		resample->SetDefaultPixelValue(0);

		const ImageType::SpacingType & spacing = origSizeImage->GetSpacing();
		const ImageType::PointType & origin = origSizeImage->GetOrigin();
		ImageType::SizeType size =
			origSizeImage->GetLargestPossibleRegion().GetSize();

		using TransformType = itk::AffineTransform<double, Dimension>;
		TransformType::Pointer transform = TransformType::New();
		TransformType::OutputVectorType translation1;
		const double imageCenterX = origin[0] + spacing[0] * size[0] / 2.0;
		const double imageCenterY = origin[1] + spacing[1] * size[1] / 2.0;
		const double imageCenterZ = origin[2] + spacing[2] * size[2] / 2.0;
		translation1[0] = -imageCenterX;
		translation1[1] = -imageCenterY;
		translation1[2] = -imageCenterZ;
		transform->Translate(translation1);



		ImageType::Pointer origImage = warpImage(origSizeImage, rotationMatrix);


		TransformType::OutputVectorType translation2;
		translation2[0] = imageCenterX;
		translation2[1] = imageCenterY;
		translation2[2] = imageCenterZ;
		transform->Translate(translation2, false);



		try
		{
			itk::WriteImage(origImage, finallyPath + int2string(4, i) + ".nii");
			std::cout << i << std::endl;
		}
		catch (itk::ExceptionObject & error)
		{
			std::cerr << "Error: " << error << std::endl;
			return EXIT_FAILURE;
		}
	}
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
	//itk  77*95*52
	//tensor  52*95*77
	//这里的size不能按照itk的imageSize
	//在python中，sitk to numpy会交换第一维和第三维
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
	
	Eigen::Matrix3f am_eigen;  //不包含三个平移参数
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_eigen(b, a) = am[a * 3 + b];
		}
	}
	Eigen::Matrix3f am_eigen_inverse = am_eigen.inverse();   //求出的逆和原矩阵相乘结果不为1？
	//std::cout << am_eigen_reverse * am_eigen << std::endl;

	Eigen::Vector3f am_trans;
	am_trans << am[9], am[10], am[11];
	Eigen::Vector3f am_trans_inverse = am_trans.transpose() * (-am_eigen_inverse);


	//std::cout << am_trans << std::endl <<std::endl<< am_trans_inverse << std::endl;
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

	//std::cout << affineMatrix.sizes() << std::endl;
	//tensor在cpu上可以转vector，在GPU上不行
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
	//文件句柄
	//注意：我发现有些文章代码此处是long类型，实测运行中会报错访问异常
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,递归查找
			//如果不是,把文件绝对路径存入vector中
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

/* 函数说明 整型转固定格式的字符串
输入：
n 需要输出的字符串长度
i 需要结构化的整型
输出：
返回转化后的字符串
*/
std::string int2string(int n, int i)
{
	char s[BUFSIZ];
	sprintf_s(s, "%d", i);
	int l = strlen(s);  // 整型的位数

	if (l > n)
	{
		std::cout << "整型的长度大于需要格式化的字符串长度！";
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
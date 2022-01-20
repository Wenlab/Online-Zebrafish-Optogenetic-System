// T2C_Test3.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// 此程序在T2C_Test2 和 MutiThreads 的基础上测试相机通信
// 此程序重点实现将 GPU Express Path 加入流程之中，同时采用多线程方法管理流程
// TODO用来标注需要实现的任务
// NEW用来标注新加入的语句


#include <iostream>
#include "atcore.h"
#include "Talk2Camera.h"
#include "CvStruct.h"
#include <time.h>
#include "SimpleAcq_utility_kernels.h"
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;


extern void ImgReverseOnGpuFunc(unsigned short* inputBuffer, unsigned short bufferWidth, unsigned short bufferHeight,
	cudaStream_t* stream);

//实时显示相机图像
void ShowCamera(CamData* cam_data, AT_H cam_handle, AT_GPU_H gpu_handle, CvStruct* cv) {//Show real-time Images
	cout << "Showing camera. Press ESC to end." << endl;

	//Define variables
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1); //临时图像, 将用于存储原始图像
	IplImage* tempImage2 = cvCreateImage(cv->ImgSize, IPL_DEPTH_8U, 1); //临时图像, 将用于存储反色图像

	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle); //开始获取,注意该函数调用之后应当尽快开始WaitBuffer操作
	Sleep(100); //等待100ms
	while (1) { //图像获取显示循环, 结束循环依赖break
		if (T2Cam_GPU_GrabFrame(cam_data, cam_handle, gpu_handle) == 0) { //获取图像并判断是否成功. TODO: 改造获取图像函数的返回方式, 返回SDK中自带的错误码
			// 需要注意的是, 不管相机设置的是什么图像格式, T2Cam_GrabFrame中给出的"原始数据"ImageRawData其实已经经过了一步转码变成了Mono16的格式, 不用担心格式问题
			double scale = static_cast<double>(cv->ScalingFactor + 1) / 100; //换算得出真实的线性变换相乘系数

			// 变量准备
			void** cpuBufferArray, ** gpuBufferArray, * streamPtr;
			int pathIndex = (cam_data->iFrameNumber - 1) % NumberOfPaths;

			// 反色处理

			AT_GPU_CopyInputCpuToInputGpu(gpu_handle, pathIndex, 0);
			AT_GPU_GetInputGpuBufferArray(gpu_handle, pathIndex, &gpuBufferArray);
			AT_GPU_GetStreamPtr(gpu_handle, pathIndex, &streamPtr);
			ImgReverseOnGpuFunc(reinterpret_cast<unsigned short*>(gpuBufferArray[0]), cam_data->ImageWidth, cam_data->ImageHeight, reinterpret_cast<cudaStream_t*>(streamPtr));

			// 将原始图像存入tempImage, 线性变换后存入cv->Image中
			AT_GPU_GetInputCpuBufferArray(gpu_handle, pathIndex, &cpuBufferArray);
			for (int i = 0; i < tempImage->height; i++) { //逐行将数据存入临时图像中
				memcpy(tempImage->imageData + i * tempImage->widthStep, \
					reinterpret_cast<unsigned short*>(cpuBufferArray[0]) + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}
			cvConvertScale(tempImage, cv->Image, scale, (cv->Shift) - 255); //将临时图像线性变换变换后存入cv->Image中
			//cvConvertScale(tempImage, cv->Image);

			// 将反色图像存入tempImage, 线性变换后存入tempImage2中
			AT_GPU_CopyInputGpuToOutputCpu(gpu_handle, pathIndex, 0);
			AT_GPU_GetOutputCpuBufferArray(gpu_handle, pathIndex, &cpuBufferArray);
			AT_GPU_WaitPath(gpu_handle, pathIndex);
			for (int i = 0; i < tempImage->height; i++) { //逐行将数据存入临时图像中
				memcpy(tempImage->imageData + i * tempImage->widthStep, \
					reinterpret_cast<unsigned short*>(cpuBufferArray[0]) + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}
			cvConvertScale(tempImage, tempImage2, scale, -(cv->Shift) + 255 - (scale - 1) * 255);
			//cvConvertScale(tempImage, tempImage2);

			//显示图像
			cvShowImage("FromCamera", cv->Image); //显示线性变换后图像
			cvShowImage("ReverseImage", tempImage2); //显示反色图像

		}
		else {
			cout << "Error in imaging acquisition!" << endl;
			break;
		}
		if (cvWaitKey(1) == 27)break;//End the loop when ESC detected
	}
	//结束获取
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;

	//Release variables
	cvReleaseImage(&tempImage); //释放临时图像
	cvReleaseImage(&tempImage2); //释放临时图像
}

//帧率测试相关函数
double fpsTest4(CamData* cam_data, AT_H cam_handle, CvStruct* cv) {//Frame rate test 4, test the frame rate after transcoding and display
	//Sets the number of captured frames
	int counttimes = 1000;
	cout << "Test Mode 4. How many frames do you want to Grab?" << endl;
	cin >> counttimes;

	//Define variables
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1);
	int count = 0;

	//Start Acquisition and timing
	clock_t start, end;
	cout << endl << "Frame Number:" << endl;
	T2Cam_StartAcquisition(cam_handle);
	Sleep(100);
	start = clock();
	while (count < counttimes) {
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0) {
			double scale = static_cast<double>(cv->ScalingFactor + 1) / 100;
			//Memory transfer
			for (int i = 0; i < tempImage->height; i++) {
				memcpy(tempImage->imageData + i * tempImage->widthStep, cam_data->ImageRawData + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}

			//Transcode and display
			cvConvertScale(tempImage, cv->Image, scale, (cv->Shift) - 255);
			cvShowImage("FromCamera", cv->Image);
			cvShowImage("ReverseImage", tempImage);

			count++;
			cout << "\r" << count;
		}
		else {
			cout << "Error in imaging acquisition! Count:" << count++ << endl;
			break;
		}
		if (cvWaitKey(1) == 27)break;
	}
	end = clock();
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;
	//End Acquisition and timing

	//Calculate the frame rate
	double timecost = (double)(end - start) / CLK_TCK;
	cout << "It costs " << timecost << "s to finish " << counttimes << " times grab." << endl;
	double fps = (double)counttimes / timecost;
	cout << "The camera has a frame rate up to" << fps << "fps." << endl;

	//Release variables
	cvReleaseImage(&tempImage);

	//Return the frame rate
	return fps;
}

double fpsTest3(CamData* cam_data, AT_H cam_handle, CvStruct* cv) {//Frame rate test 3, Test the frame rate when the original image is displayed
	//Sets the number of captured frames
	int counttimes = 1000;
	cout << "Test Mode 3. How many frames do you want to Grab?" << endl;
	cin >> counttimes;

	//Define variables
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1);
	int count = 0;

	//Start Acquisition and timing
	clock_t start, end;
	cout << endl << "Frame Number:" << endl;
	T2Cam_StartAcquisition(cam_handle);
	Sleep(100);
	start = clock();
	while (count < counttimes) {
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0) {
			//Memory transfer
			for (int i = 0; i < tempImage->height; i++) {
				memcpy(tempImage->imageData + i * tempImage->widthStep, cam_data->ImageRawData + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}

			cvShowImage("ReverseImage", tempImage);
			count++;
			cout << "\r" << count;
		}
		else {
			cout << "Error in imaging acquisition! Count:" << count++ << endl;
			break;
		}
		if (cvWaitKey(1) == 27)break;
	}
	end = clock();
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;
	//End Acquisition and timing

	//Calculate the frame rate
	double timecost = (double)(end - start) / CLK_TCK;
	cout << "It costs " << timecost << "s to finish " << counttimes << " times grab." << endl;
	double fps = (double)counttimes / timecost;
	cout << "The camera has a frame rate up to" << fps << "fps." << endl;

	//Release variables
	cvReleaseImage(&tempImage);

	//Return the frame rate
	return fps;
}

double fpsTest2(CamData* cam_data, AT_H cam_handle, CvStruct* cv) {//Frame rate test 2, tests the frame rate when the original image is only transmitted but not displayed
	//Sets the number of captured frames
	int counttimes = 1000;
	cout << "Test Mode 2. How many frames do you want to Grab?" << endl;
	cin >> counttimes;

	//Define variables
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1);
	int count = 0;

	//Start Acquisition and timing
	clock_t start, end;
	cout << endl << "Frame Number:" << endl;
	T2Cam_StartAcquisition(cam_handle);
	Sleep(100);
	start = clock();
	while (count < counttimes) {
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0) {
			//Memory transfer
			for (int i = 0; i < tempImage->height; i++) {
				memcpy(tempImage->imageData + i * (tempImage->widthStep), cam_data->ImageRawData + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}
			count++;
			cout << "\r" << count;
		}
		else {
			cout << "Error in imaging acquisition! Count:" << count++ << endl;
			break;
		}
	}
	end = clock();
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;
	//End Acquisition and timing

	//Calculate the frame rate
	double timecost = (double)(end - start) / CLK_TCK;
	cout << "It costs " << timecost << "s to finish " << counttimes << " times grab." << endl;
	double fps = (double)counttimes / timecost;
	cout << "The camera has a frame rate up to" << fps << "fps." << endl;

	//Release variables
	cvReleaseImage(&tempImage);
	cout << "Release" << endl;
	//Return the frame rate
	return fps;
}

double fpsTest1(CamData* cam_data, AT_H cam_handle) {//Frame rate test 1, tests the frame rate when the image is only captured
	//Sets the number of captured frames
	int counttimes = 1000;
	cout << "Test Mode 1. How many frames do you want to Grab?" << endl;
	cin >> counttimes;
	//Define variables
	int count = 0;

	//Start Acquisition and timing
	clock_t start, end;
	cout << endl << "Frame Number:" << endl;
	T2Cam_StartAcquisition(cam_handle);
	Sleep(100);
	start = clock();
	while (count < counttimes) {
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0) {
			count++;
			cout << "\r" << count;
		}
		else {
			cout << "Error in imaging acquisition! Count:" << count++ << endl;
			break;
		}
	}
	end = clock();
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;
	//End Acquisition and timing

	//Calculate the frame rate
	double timecost = (double)(end - start) / CLK_TCK;
	cout << "It costs " << timecost << "s to finish " << counttimes << " times grab." << endl;
	double fps = (double)counttimes / timecost;
	cout << "The camera has a frame rate up to" << fps << "fps." << endl;

	//Return the frame rate
	return fps;
}

int oldmain() {
	unsigned short test1 = 5;
	cout << sizeof(test1) << endl;

	bool Continue = true; //该变量用于标识是否再进行一轮测试
	while (Continue) {//Loop for the whole test, 测试主循环
	//相机和AndorSDK3的初始化与设置
	//Initialize the Camera and Andor libs

		//变量准备
		AT_H cam_handle; //相机句柄变量的声明,随后将再初始化时为其赋值,之后用它来传递相机信息
		CamData* cam_data = T2Cam_CreateCamData(); //动态申请CamData结构体的空间,创建指向该空间的cam_data指针
		AT_GPU_H gpu_handle; //NEW: GPU Express的句柄


		//自动设置选择
		bool Auto = false;
		cout << "Do you want to Set Up Automatically? 0 or 1" << endl;
		cin >> Auto;

		//开始初始化
		//TODO: 这里的两种初始化没必要一个一个函数调用, 改写Talk2Camara.cpp, 用结构体传递各设置参数
		//  一个函数用于向用户询问设置结构体的各项参数
		//  另一个函数用于将结构体中各个参数赋予相机
		//TODO: 之后将会把GPU Exp Path的初始化加入
		if (Auto == true) {
			//自动初始化
			T2Cam_AutoInitialize(cam_data, &cam_handle);
		}
		else {
			T2Cam_InitializeLib(&cam_handle);

			SetupBinningandAOI(cam_handle);

			T2Cam_InitializeCamData(cam_data, cam_handle);

			getUserSettings(cam_handle);

			CreateBuffer(cam_data, cam_handle);

			InitGpuExpLib(&gpu_handle); //NEW: 初始化GPU Exp

			CreateGPUExpBuffer(cam_data, gpu_handle); //NEW: 初始化GPU Path Buffer

		}

		//Open CV初始化, 创建Window与Tarkbar控件
		//allocate Image and Open CV Window
		CvStruct* cv = CreatCvStruct(cam_data);

		cvNamedWindow("FromCamera", CV_WINDOW_AUTOSIZE);
		cvNamedWindow("ReverseImage", CV_WINDOW_AUTOSIZE);
		cvCreateTrackbar("PixelScaling", "FromCamera", &(cv->ScalingFactor), 255, NULL);
		cvCreateTrackbar("PixelShift", "FromCamera", &(cv->Shift), 510, NULL);


		//获取图像并测试帧率
			//cout << "Showing camera..." << endl <<"Press any key to get next image immediately, press ESC to stop." << endl;
			//Start Grab Frames and timing
		int testmode = 5; //用于指示测试模式的变量
		while (testmode) {//测试大循环, 用于控制完整的一整次测试, 获取图像处理显示的小循环交到了switch选择的函数手中
			switch (testmode)
			{
			case 5:
				ShowCamera(cam_data, cam_handle, gpu_handle, cv);
				break;
			case 4:
				fpsTest4(cam_data, cam_handle, cv);
				break;
			case 3:
				fpsTest3(cam_data, cam_handle, cv);
				break;
			case 2:
				fpsTest2(cam_data, cam_handle, cv);
				break;
			default:
				fpsTest1(cam_data, cam_handle);
				break;
			}
			//询问下一次测试做什么的菜单
			//Set the Grab mode for next loop
			cout << "Please select a mode to measure frame rate:\n"
				<< "---0:Exit\n"
				<< "---1:Only grab frames(default)\n"
				<< "---2:Execute memcpy\n"
				<< "---3:Show RAW image\n"
				<< "---4:Show converted image\n"
				<< "---5:Just show camera, do not measure\n";
			cin >> testmode;
		}

		//结束测试的收尾工作
		system("pause");

		//关闭Open Cv, 相机, SDK, 释放内存
		//Close CV, Camera and Libs
		DestroyCvStruct(cv);
		T2Cam_TurnOff(cam_data, cam_handle);
		CloseGPUExpLib(gpu_handle); //NEW: 关闭GPU Exp
		T2Cam_CloseLib();

		// 选择是否进行下一次测试主循环
		//Choose whether to continue
		cout << "Do you want to continue testing? 0 or 1\n";
		cin >> Continue;
		Auto = false;
	}
}

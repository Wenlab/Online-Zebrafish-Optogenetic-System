// T2C_Test3.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// 此程序在T2C_Test2 和 MutiThreads 的基础上测试相机通信
// 此程序重点实现将 GPU Express Path 加入流程之中，同时采用多线程方法管理流程
// TODO用来标注需要实现的任务
// NEW用来标注新加入的语句


#include <iostream>
#include "atcore.h"
#include "Talk2Camera.h"
#include <time.h>

#include <cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;


//帧率测试相关函数
double fpsTest4(CamData* cam_data, AT_H cam_handle) 
{
	//Frame rate test 4, test the frame rate after transcoding and display
	//Sets the number of captured frames
	int counttimes = 1000;
	cout << "Test Mode 4. How many frames do you want to Grab?" << endl;
	cin >> counttimes;

	//Define variables
	//IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1);
	cv::Mat saveImage(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1);
	cv::Mat showImgRescale(cv::Size(CCDSIZEX / 4, CCDSIZEY / 4), CV_16UC1);
	int count = 0;

	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle);
	while (count < counttimes) 
	{
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0)
		{
			memcpy(saveImage.data, cam_data->ImageRawData, saveImage.cols *saveImage.rows * sizeof(unsigned short));
			flip(saveImage, saveImage, 0);   //flipcode=0,垂直翻转图像
			imwrite("D:/kexin/T2C_Test3/testSave/" + to_string(count) + ".tif", saveImage, std::vector<int>{259, 1}); //保存图片  //TIFFTAG_COMPRESSION=259 COMPRESSION_NONE=1  保存tif格式时不要压缩

			//显示图像，自动拉对比度
			cv::resize(saveImage, showImgRescale, cv::Size(CCDSIZEX / 4, CCDSIZEY / 4));
			double m = 0, M = 0;
			cv::minMaxLoc(showImgRescale, &m, &M);
			showImgRescale = 255 * (showImgRescale - m) / (M - m);
			showImgRescale.convertTo(showImgRescale, CV_8UC1);
			cv::imshow("FromCamera", showImgRescale); //显示改变过对比度的图像(储存的图像)

			count++;
			cout << "\r" << count;
		}
		else {
			cout << "Error in imaging acquisition! Count:" << count++ << endl;
			break;
		}
		if (cvWaitKey(1) == 27)break;
	}
	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;

	return 0;
}


int main() 
{
	//Loop for the whole test, 测试主循环
	//相机和AndorSDK3的初始化与设置
	//Initialize the Camera and Andor libs

	//变量准备
	AT_H cam_handle; //相机句柄变量的声明,随后将再初始化时为其赋值,之后用它来传递相机信息
	CamData* cam_data = T2Cam_CreateCamData(); //动态申请CamData结构体的空间,创建指向该空间的cam_data指针


	//开始初始化
	T2Cam_InitializeLib(&cam_handle);
	SetupBinningandAOI(cam_handle);
	T2Cam_InitializeCamData(cam_data, cam_handle);
	getUserSettings(cam_handle);
	CreateBuffer(cam_data, cam_handle);

	fpsTest4(cam_data, cam_handle);

	//结束测试的收尾工作
	system("pause");

	//相机, SDK, 释放内存
	//Camera and Libs
	T2Cam_TurnOff(cam_data, cam_handle);
	T2Cam_CloseLib();

	return 0;
}

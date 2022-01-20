#include "DisplayImg.h"
#include <iostream>
#include <sstream>
using namespace std;

void DisplayImg(AT_GPU_H gpu_handle, CvStruct* cv, Locks* locks) {
	cout << "Showing camera. Press ESC to end.(Display)" << endl;
	
	// 变量准备 //Define variables
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1); //临时图像, 将用于存储原始图像
	cv::Mat saveImage(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1);
	cv::Mat showImgRescale(cv::Size(CCDSIZEX / 4, CCDSIZEY / 4), CV_16UC1);
	void** cpuBufferArray;
	unsigned long long frameNumber = 0;
	int bufferIndex = 0, pathIndex = 0, bufferIndexInPath = 0;
	bool* flag_end = &(locks->flag_end);
	ostringstream sbuffer;
	//创建Window与Tarkbar控件
	cvNamedWindow("FromCamera", CV_WINDOW_AUTOSIZE);
	cvCreateTrackbar("PixelScaling", "FromCamera", &(cv->ScalingFactor), 255, NULL);
	cvCreateTrackbar("PixelShift", "FromCamera", &(cv->Shift), 510, NULL);

	while (!(*flag_end)) { //图像显示循环
		// 需要注意的是, 不管相机设置的是什么图像格式, T2Cam_GrabFrame中给出的"原始数据"ImageRawData其实已经经过了一步转码变成了Mono16的格式, 不用担心格式问题
		unique_lock<mutex> lock_disp(locks->mutex_disp[bufferIndex]); //加锁必须在第一步, 利用互斥锁获得的优先效应尽量防止获取线程插空"超车"
		while (!(locks->flag_disp[bufferIndex])) {
			locks->cond_disp[bufferIndex].wait(lock_disp);
		}
		
		locks->flag_disp[bufferIndex] = false;
		//这种互斥锁-条件变量-bool变量三重组合最大限度保证了线程同步的健壮性
		//如果处理指令早已发出, 则加锁后不会经过while循环触发条件变量的等待, 程序正常运行
		//如果处理指令还没有发出, 则加锁后进入while循环触发条件变量等待, 期间锁被解开防止阻塞其他程序
		//同时, while检查了bool变量, 防止虚假唤醒
		if (!(*flag_end)) {
			double scale = static_cast<double>(cv->ScalingFactor + 1) / 100; //换算得出真实的线性变换相乘系数

			// 将原始图像存入tempImage, 线性变换后存入cv->Image中
			AT_GPU_GetInputCpuBufferArray(gpu_handle, pathIndex, &cpuBufferArray);
			for (int i = 0; i < tempImage->height; i++) { //逐行将数据存入临时图像中
				memcpy(tempImage->imageData + i * tempImage->widthStep,
					reinterpret_cast<unsigned short*>(cpuBufferArray[bufferIndexInPath]) + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}
			//cvConvertScale(tempImage, cv->Image, scale, (cv->Shift) - 255); //将临时图像线性变换变换后存入cv->Image中
			//cvConvertScale(tempImage, cv->Image);


			//存储图像   
			saveImage = cv::cvarrToMat(tempImage);
			flip(saveImage, saveImage, 0);   //flipcode=0,垂直翻转图像
			sbuffer.str(""); //清空字符串流
			sbuffer << SAVEPATH << "frame[" << frameNumber << "]_path[" << frameNumber / BuffersPerPath << "]_buffer_[" << bufferIndexInPath << "].tif"; //编辑文件名称
			imwrite(sbuffer.str(), saveImage, std::vector<int>{259,1}); //保存图片  //TIFFTAG_COMPRESSION=259 COMPRESSION_NONE=1  保存tif格式时不要压缩
			
			//saveImage = cv::cvarrToMat(tempImage);
			//double min = NULL;
			//double max = NULL;
			//cv::minMaxIdx(saveImage, &min, &max);
			//cout << "min: " << min << "    max: " << max << endl;

			//调整各编号
			frameNumber++;
			bufferIndex = frameNumber % NumberOfBuffers;
			pathIndex = (frameNumber / BuffersPerPath) % NumberOfPaths;
			bufferIndexInPath = frameNumber % BuffersPerPath;



			//显示图像
			cv::resize(saveImage, showImgRescale, cv::Size(CCDSIZEX / 4, CCDSIZEY / 4));
			double m = 0, M = 0;
			cv::minMaxLoc(showImgRescale, &m, &M);
			showImgRescale = 255 * (showImgRescale - m) / (M - m);
			showImgRescale.convertTo(showImgRescale, CV_8UC1);
			//cout << m << "    " << M << endl;
			cv::imshow("FromCamera", showImgRescale); //显示原图(储存的图像)

			if (cvWaitKey(1) == 27) {//没有这一句图像无法显示，所以也就顺便让它监测停止信号了
				*flag_end = true;
			}
		}
	}

	//Release variables
	cvReleaseImage(&tempImage); //释放临时图像
	return;
}
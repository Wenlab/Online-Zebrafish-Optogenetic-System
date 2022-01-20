#include "Output.h"
using namespace std;

void Output(AT_GPU_H gpu_handle, CvStruct* cv,Locks* locks) {
	//变量准备
	IplImage* tempImage = cvCreateImage(cv->ImgSize, IPL_DEPTH_16U, 1); //临时图像, 将用于存储原始图像
	IplImage* tempImage2 = cvCreateImage(cv->ImgSize, IPL_DEPTH_8U, 1); //临时图像, 将用于存储反色图像
	void** cpuBufferArray;
	unsigned long long frameNumber = 0;
	int  pathIndex = 0, bufferIndexInPath;
	bool* flag_end = &(locks->flag_end);
	//cvNamedWindow("ReverseImage", CV_WINDOW_AUTOSIZE);

	while (!(*flag_end)) { //输出处理主循环
		//加锁
		unique_lock<mutex> lock_output(locks->mutex_output[pathIndex]); //加锁必须在第一步, 利用互斥锁获得的优先效应尽量防止获取线程插空"超车"
		while (!(locks->flag_output[pathIndex])) {
			locks->cond_output[pathIndex].wait(lock_output);
		}
		locks->flag_output[pathIndex] = false;
		//这种互斥锁-条件变量-bool变量三重组合最大限度保证了线程同步的健壮性
		//如果处理指令早已发出, 则加锁后不会经过while循环触发条件变量的等待, 程序正常运行
		//如果处理指令还没有发出, 则加锁后进入while循环触发条件变量等待, 期间锁被解开防止阻塞其他程序
		//同时, while检查了bool变量, 防止虚假唤醒
		if (!(*flag_end)) {
			double scale = static_cast<double>(cv->ScalingFactor + 1) / 100; //换算得出真实的线性变换相乘系数

			// 将反色图像存入tempImage, 线性变换后存入tempImage2中
			AT_GPU_CopyInputGpuToOutputCpu(gpu_handle, pathIndex, 0);
			AT_GPU_GetOutputCpuBufferArray(gpu_handle, pathIndex, &cpuBufferArray);
			AT_GPU_WaitPath(gpu_handle, pathIndex);
			for (int i = 0; i < tempImage->height; i++) { //逐行将数据存入临时图像中
				memcpy(tempImage->imageData + i * tempImage->widthStep,
					reinterpret_cast<unsigned short*>(cpuBufferArray[0]) + i * tempImage->width, tempImage->width * sizeof(unsigned short));
			}
			//cvConvertScale(tempImage, tempImage2, scale, -(cv->Shift) + 255 - (scale - 1) * 255);
			//cvConvertScale(tempImage, tempImage2);

			//显示图像
			//cvShowImage("ReverseImage", tempImage); //显示反色图像
			if (cvWaitKey(1) == 27) {
				*flag_end = true;
			}
		}
	}
	cvReleaseImage(&tempImage); //释放临时图像
	cvReleaseImage(&tempImage2); //释放临时图像
	return;
}
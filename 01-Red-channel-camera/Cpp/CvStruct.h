#pragma once

#include <cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Talk2Camera.h"

//定义用于Cv显示的结构体 CvStruct
typedef struct CvStruct {
	IplImage* Image; //指向图像结构体的指针, 指向当前展示的图像结构体
	// 需要注意的是, 不管相机设置的是什么图像格式, T2Cam_GrabFrame中给出的"原始数据"ImageRawData其实已经经过了一步转码变成了Mono16的格式, 不用担心格式问题
	CvSize ImgSize; //图像大小结构体(由opencv定义,有height和width两个int成员)

	int ScalingFactor; //线性变换相乘系数的滑块值(受openCv限定为int且从0开始), 在使用时换算为真实值(double scale = static_cast<double>(cv->ScalingFactor + 1) / 100;)
	int Shift; //线性变换相加系数的滑块值(受openCv限定为int且从0开始), 在使用时换算为真实值((cv->Shift) - 255)
}CvStruct;

CvStruct* CreatCvStruct(CamData* cam_data);

void DestroyCvStruct(CvStruct* cv);
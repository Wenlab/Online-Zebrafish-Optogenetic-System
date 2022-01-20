#include "CvStruct.h"
#include <iostream>
using std::cout;

//使用相机数据,创建CvStruct,返回其指针
//利用相机设置数据, 在相机初始化与设置完成后调用
//创建的CvStruct一方面用存储了图像直接与OpenCv对接的相关参数, 另一方面存储用于当前展示的图像数据
CvStruct* CreatCvStruct(CamData* cam_data) {
	CvStruct* cv = (CvStruct*)malloc(sizeof(CvStruct)); //向内存动态申请空间,记得free

	cv->ImgSize.width = (int)cam_data->ImageWidth; //图像宽
	cv->ImgSize.height = (int)cam_data->ImageHeight; //图像高
	cv->Image = cvCreateImage(cv->ImgSize, IPL_DEPTH_8U, 1); //图像结构体创建,动态申请,记得ReleaseImage
	// 需要注意的是, 不管相机设置的是什么图像格式, T2Cam_GrabFrame中给出的"原始数据"ImageRawData其实已经经过了一步转码变成了Mono16的格式, 不用担心格式问题
	cv->ScalingFactor = 40; //线性变换相乘系数
	cv->Shift = 255; //线性变换相加系数

	return cv; //返回指向CvStruct的指针
}

//销毁CvStruct. CvStruct和其Image成员在创建时是动态申请的空间,需要手动释放
void DestroyCvStruct(CvStruct* cv) {

	cout << "DestroyCvStruct\n";
	cvReleaseImage(&(cv->Image)); //销毁Image成员
	cout << "ReleaseImage\n";
	free(cv); // 销毁CvStruct
	cout << "freeCV\n";
	return;
}
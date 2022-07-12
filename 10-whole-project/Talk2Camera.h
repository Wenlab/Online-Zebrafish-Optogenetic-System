/*
 * Talk2Camera.h
 *
 *  Created on: Sep 3, 2013
 *      Author: quan Wen, modified from Andy's code
 */

//TODO: 统一设置参数的方法, 增强其可扩展性
//	固定参数设置: 初始化时有一些参数是固定的,不给用户更改的权限,这些东西用宏定义
//	可变参数设置: 定义获取-修改-反馈三个函数, 在此基础上定义可变参数设置函数, 使得每个参数的设置只需一句话
//		出错依旧使用返回值判断, 返回值就利用Andor自带的错误码就行了, 只不过输出错误信息的语句写在"反馈"函数中, 无需每次都写一遍

//图像格式问题: 
//	位深: 图像的位深是指一个像素占多少个bit, 这些bit全0和全1为黑或白, 中间值表示不同灰度. 位深越大, 对明暗的区分更精细.
//	Zyla相机支持四种图像格式:
//		Mono12: 12位深灰度图, 用2Byte存储一个像素, 缺的4个bit用0填充
//		Mono12Packed: 12位深灰度图, 用3Byte存储相邻两个像素, 刚好1个像素占12bit
//		Mono16: 16位深灰度图
//		Mono32: 32位深灰度图
//	为了提高速度, 要减小获取图像的数据量, 自然使用12位深灰度图; 要减小传输的数据量, 自然要用Mono12Packed来传输
//	拿到的Mono12Packed数据并不能直接对接OpenCv, 需要先使用AT_ConvertBuffer函数转化为Mono16再交给OpenCv的IplImage结构体
//	而不同的IplImage结构体之间可以在cvConvertScale时自动完成转码
//	在现有的程序中, CvStruct最后会拿到8位灰度图, 因为CvStruct中的Image是用8位灰度图形式创建的IplImage结构体
#pragma once
#ifndef TALK2CAMERA_H_
#define TALK2CAMERA_H_

#include <atcore.h>
#include <stdio.h>

#define PRINT_DEBUG 0
#define NumberOfBuffers 20 //用于获取图像的Buffer数量, 若启用GpuExpPath, 应有 NumberOfBuffers >= NumberOfPaths*BuffersPerPath
#define NumberOfPaths 4 //GpuExpPath的数量
#define BuffersPerPath 1 //每一个Path中存储的图像数

/*
 * We define a new variable type, "CamData" which is a struct that
 * holds information about the current Data in the camera.
 *
 * The actual data resides in *iImageData
 * The i notation indicates that these are internal values. e.g.
 * iFrameNumber refers to the FrameNumber that the camera sees,
 *
 */
typedef struct CamDataStruct CamData;

struct CamDataStruct {


	AT_64 ImageHeight; //图像高度(像素数)
	AT_64 ImageWidth; //图像宽度(像素数)
	AT_64 ImageStride; //(转码前)一行图像的大小(Byte数),由于一行图像末尾或有额外填充的padding区域,所以这个参数并不等于 ImageWidth*每个像素的Byte数
	AT_64 ImageSizeBytes; //(转码前)图像大小)
	AT_WC PixelEncoding[64]; //(转码前)编码格式(字符数组)
	unsigned long long iFrameNumber; //帧数, 每次获取一帧转码完毕并且将缓冲区重新入队后++, unsigned long long的类型保证了无法溢出 (100fps的话, 需要5.8亿年才会溢出)
	unsigned short* ImageRawData; //指向转码后图像的指针, 由于Mono16格式和short格式都是一个数据占2字节, 所以这里使用unsigned short指针
									//大小: ImageHeight*ImageWidth*2Byte, 申请空间时直接申请元素个数为 ImageHeight*ImageWidth 的short数组
									//注意, 由于转码会清除padding区域, 转码后图像大小即为 ImageWidth*ImageHeight*每个像素的Byte数, 但是转码前的原始图像存在padding区域, 所以二者大小并不能简单的用两种格式每个像素的Byte数来换算
	unsigned char* AcqBuffers[NumberOfBuffers]; //指向申请的原始图像存储空间的的指针数组, 单个元素指向的存储空间大小为 ImageSizeBytes + 7
	unsigned char* AlignedBuffers[NumberOfBuffers]; //指向调整后的的原始图像存储空间的的指针数组, 单个元素指向的存储空间大小为 ImageSizeBytes
	};

/*
 * Rows and pixels of camera
 */
#define CCDSIZEX 2048
#define CCDSIZEY 2048
#define CCDENCODING L"Mono16" //L"Mono12Packed" or L"Mono12" or L"Mono16"

/*
 * Initalizes the library and provides the  license key for
 * the Imaging control software. The function returns a
 * non-zero value if successful.
 */
int T2Cam_InitializeLib(int* Hndl);

/*
 * Closes the library.
 *
 */
void T2Cam_CloseLib();



/*
 * Create CamData type, this function will allocate
 * memory for raw image data.
 */

CamData* T2Cam_CreateCamData();

void T2Cam_InitializeCamData(CamData* MyCamera,int _handle);


int T2Cam_GrabFrame(CamData* MyCamera, int _handle);

void T2Cam_TurnOff(CamData* MyCamera,int _handle);

void SetupSensorCooling(int _handle);

int CreateBuffer(CamData* MyCamera,int _handle);

void T2Cam_StartAcquisition(int _handle);

int getUserSettings(int _handle);

int SetupBinningandAOI(int _handle);

void deleteBuffers(CamData* MyCamera);


int AutogetUserSettings(int _handle);


void T2Cam_Close(CamData* MyCamera, AT_H _handle);


#endif /* TALK2CAMERA_H_ */

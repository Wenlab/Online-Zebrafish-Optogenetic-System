#include "GrabImg.h"
#include "DisplayImg.h"
#include "Output.h"
#include <thread>
#include <iostream>
using namespace std;

void Init(CamData** cam_data, AT_H* cam_handle, AT_GPU_H* gpu_handle, CvStruct** cv, Locks** locks);

void Close(CamData* cam_data, AT_H cam_handle, AT_GPU_H gpu_handle, CvStruct* cv, Locks* locks);

int main() {
	//变量准备
	CamData* cam_data; //动态申请CamData结构体的空间,创建指向该空间的cam_data指针
	AT_H cam_handle; //相机句柄变量的声明,随后将再初始化时为其赋值,之后用它来传递相机信息
	AT_GPU_H gpu_handle; //GPU Express的句柄
	CvStruct* cv;
	Locks* locks;

	//初始化
	Init(&cam_data, &cam_handle, &gpu_handle, &cv, &locks);
		
	//创建线程, 开始处理
	thread Display(DisplayImg, gpu_handle, cv, locks);
	thread Output(Output, gpu_handle, cv, locks);
	thread Grab(GrabImg, cam_data, cam_handle, gpu_handle, locks);

	//等待线程结束
	Display.join();
	Output.join();
	Grab.join();

	//结束收尾
	Close(cam_data, cam_handle, gpu_handle, cv, locks);
}

void Init(CamData** cam_data, AT_H* cam_handle, AT_GPU_H* gpu_handle, CvStruct** cv, Locks** locks) {
	//相机数据结构体创建
	*cam_data = T2Cam_CreateCamData();
	//相机和AndorSDK3的初始化与设置 //Initialize the Camera and Andor libs
	T2Cam_GPU_Initialize(*cam_data, cam_handle, gpu_handle);
	//Locks结构体初始化 
	*locks = CreatLocks();
	//CvStruct结构体初始化
	*cv = CreatCvStruct(*cam_data);
}

void Close(CamData* cam_data, AT_H cam_handle, AT_GPU_H gpu_handle, CvStruct* cv, Locks* locks) {
	//关闭Open Cv, 相机, SDK, 释放内存 //Close CV, Camera and Libs
	DestroyLocks(locks);
	DestroyCvStruct(cv);
	T2Cam_TurnOff(cam_data, cam_handle);
	CloseGPUExpLib(gpu_handle);
	T2Cam_CloseLib();
}
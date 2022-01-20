#include "GrabImg.h"
#include <iostream>
#include <time.h>
using namespace std;

void GrabImg(CamData* cam_data, AT_H cam_handle, AT_GPU_H gpu_handle, Locks* locks) {//Show real-time Images
	cout << "Showing camera. Press ESC to end.(Grab)" << endl;

	// 变量准备
	void** cpuBufferArray, ** gpuBufferArray, * streamPtr;
	unsigned int pathIndex, bufferIndex, bufferIndexInPath; // 用于存储path, buffer, path中的buffer下标. 循环赋值, 无溢出风险.
	bool flag_AcqError = false; //用于标记获取失败
	clock_t start, end; //获取帧率计时变量
	unsigned long long batchNum = 0, batchStart = 0;//用于存储批数和开始帧数, 帧数相关变量, 有溢出风险, 故而使用 long long
	unsigned int counttimes; //用于计算一批的帧数
	double timecost, fps; //用于计算一批用时与帧率
	bool* flag_end = &(locks->flag_end);
	//cout << "Grab 1!" << endl;
	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle); //开始获取,注意该函数调用之后应当尽快开始WaitBuffer操作
	Sleep(100); //等待100ms
	start = clock();
	while (!(*flag_end)) { //图像获取显示循环, 结束循环依赖break
		for (bufferIndexInPath = 0; bufferIndexInPath < BuffersPerPath; bufferIndexInPath++) {
			//获得本次操作对应的path和buffer编号
			pathIndex = (cam_data->iFrameNumber / BuffersPerPath) % NumberOfPaths; //图像对应的Path编号
			bufferIndex = cam_data->iFrameNumber % NumberOfBuffers; //图像对应的Buffer编号
			//cout << "Grab 2!" << endl;
			//加锁: 由于存储和显示均需要访问已有的图像, 所以在存储显示运行期间不能获取对应buffer的图像, 若此处加锁失败, 说明存储显示"追尾"
			unique_lock<mutex> lock_disp(locks->mutex_disp[bufferIndex]);
			//cout << "Grab 3!" << endl;
			if (T2Cam_GPU_GrabFrame(cam_data, cam_handle, gpu_handle) == 0) { //获取图像并判断是否成功. TODO: 改造获取图像函数的返回方式, 返回SDK中自带的错误码
				//需要注意的是, 不管相机设置的是什么图像格式, T2Cam_GrabFrame中给出的"原始数据"ImageRawData其实已经经过了一步转码变成了Mono16的格式, 不用担心格式问题
				lock_disp.unlock(); locks->flag_disp[bufferIndex] = true; locks->cond_disp[bufferIndex].notify_one(); //唤醒disp线程
				//cout << "Grab 4!" << endl;
			}
			else {
				cout << "Error in imaging acquisition!" << endl;
				*flag_end = true;
				flag_AcqError = true;
				break;
			}
		}
		if (!flag_AcqError) { //若一批图像获取成功, 则向GPU发出拷贝处理指令
			//加锁: 此处假设从内存向显存的拷贝不会"追尾", 所以在获取图像的时候该buffer的图像必然不会有向对应显存的拷贝, 所以这个锁放在了获取之后
			//	由于若内存向显存的拷贝是获取完一批图像后的第一步, 这步出现"追尾", 拷贝占的时间就太长了, 这个流程也就必然不可能实现了
			unique_lock<mutex> lock_output(locks->mutex_output[pathIndex]);
			//GPU处理
			for (bufferIndexInPath = 0; bufferIndexInPath < BuffersPerPath; bufferIndexInPath++) {
				AT_GPU_CopyInputCpuToInputGpu(gpu_handle, pathIndex, bufferIndexInPath); //传入显存
			}
			AT_GPU_GetInputGpuBufferArray(gpu_handle, pathIndex, &gpuBufferArray);
			AT_GPU_GetStreamPtr(gpu_handle, pathIndex, &streamPtr);

			//GPU图像接口在这里   gpuBufferArray


			//TODO: 改造为Callback函数, 同时按照一批图像的方式更改反色方法
			ImgReverseOnGpuFunc(reinterpret_cast<unsigned short*>(gpuBufferArray[0]), 
				cam_data->ImageWidth, cam_data->ImageHeight, reinterpret_cast<cudaStream_t*>(streamPtr)); // GPU处理计算




			for (bufferIndexInPath = 0; bufferIndexInPath < BuffersPerPath; bufferIndexInPath++) {
				AT_GPU_CopyInputGpuToOutputCpu(gpu_handle, pathIndex, bufferIndexInPath); //传出显存, 视最终结果存储方式的不同更改此句
			}
			lock_output.unlock(); locks->flag_output[pathIndex] = true; locks->cond_output[pathIndex].notify_one(); //唤醒output线程
			if (cam_data->iFrameNumber / BuffersPerBatch > batchNum) { //若现在的batch数目大于存储的batchNum, 则认为完成了一个完整batch的采集
				counttimes = cam_data->iFrameNumber - batchStart; //计算一批的帧数
				batchStart = cam_data->iFrameNumber; //更新存储的开始帧数
				batchNum = cam_data->iFrameNumber / BuffersPerBatch; //更新存储的batchNum
				end = clock();
				timecost = (double)(end - start) / CLK_TCK; //计算一批的时间
				cout << "It costs " << timecost << "s to finish " << counttimes << " times grab." << endl;
				fps = (double)counttimes / timecost; //计算帧率
				cout << "The camera has a frame rate up to" << fps << "fps." << endl;
				start = clock();
			}
		}
	}
	//结束获取
	AT_Command(cam_handle, L"AcquisitionStop");
	NotifyAllLocks(locks);
	cout << endl << endl;
	return;
}
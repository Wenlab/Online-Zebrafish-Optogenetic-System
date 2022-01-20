#pragma once
#include <mutex>
#include <condition_variable>
#include "Talk2Camera.h"
using std::mutex;
using std::condition_variable;

//定义Locks结构体, 存储各个线程同步所需的互斥锁和条件变量
typedef struct Locks {
	mutex mutex_disp[NumberOfBuffers], mutex_output[NumberOfPaths]; //互斥锁数组
	condition_variable cond_disp[NumberOfBuffers], cond_output[NumberOfPaths]; //条件变量数组
	bool flag_disp[NumberOfBuffers], flag_output[NumberOfPaths], flag_end; //flag标志数组
}Locks;

Locks* CreatLocks();

void DestroyLocks(Locks* locks);

void NotifyAllLocks(Locks* locks);
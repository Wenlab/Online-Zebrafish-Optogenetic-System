#include "Locks.h"
using namespace std;

Locks* CreatLocks() {
	Locks* locks = new Locks; //向内存动态申请空间,由于结构体中存有对象,使用new申请空间,记得delete
	locks->flag_end = false;
	for (int i = 0; i < NumberOfBuffers; i++) {
		locks->flag_disp[i] = false;
	}
	for (int i = 0; i < NumberOfPaths; i++) {
		locks->flag_output[i] = false;
	}
	return locks; //返回指向Locks的指针
}

void DestroyLocks(Locks* locks) {
	delete locks; //销毁locks
	return;
}

void NotifyAllLocks(Locks* locks) {
	for (int i = 0; i < NumberOfBuffers; i++) {
		locks->flag_disp[i] = true; locks->cond_disp[i].notify_one();
	}
	for (int i = 0; i < NumberOfPaths; i++) {
		locks->flag_output[i] = true; locks->cond_output[i].notify_one();
	}
}
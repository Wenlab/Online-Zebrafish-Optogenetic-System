#pragma once
#include "CvStruct.h"
#include "Talk2Camera.h"
#include "Locks.h"

#define SAVEPATH "D:\\RealTimeOptogenetic\\testCameraSave\\"

void DisplayImg(AT_GPU_H gpu_handle, CvStruct* cv, Locks* locks);
#pragma once
#include "Talk2Camera.h"
#include "Locks.h"
#include "SimpleAcq_utility_kernels.h"

#define BuffersPerBatch 100 //用于帧率计数的一批图像的数量的下限

void GrabImg(CamData* cam_data, AT_H cam_handle, AT_GPU_H gpu_handle, Locks* locks);
#pragma once
//libtorch
#include <torch/torch.h>
#include <torch/script.h>

#include <vector>
#include <string>



//crop后图像的大小
//#define nImgSizeX 76
//#define nImgSizeY 95
//#define bandcount 50

std::vector<float> rescaleAffineMatrix(std::vector<float> v);
std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix);
//图像数据是16bit， unsigned short int
//强制转化成float， 32bit
float* readImgFromFile(std::string filename);
torch::Tensor normalizeTensor(torch::Tensor tensor);   //将tensor归一化为0~1
void saveAndCheckImage(float* imageData, int col_total, int row_total, int z_total, std::string name);   //存储图像为tiff格式
void getFileNames(std::string path, std::vector<std::string>& files);   //获取文件夹下所有文件名
std::string int2string(int n, int i);  //数字转字符，指定长度，如0001
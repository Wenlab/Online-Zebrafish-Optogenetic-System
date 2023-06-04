#pragma once
//libtorch
#include <torch/torch.h>
#include <torch/script.h>

#include <vector>
#include <string>

#include<opencv2/opencv.hpp>



//The size of the image after crop
//#define nImgSizeX 76
//#define nImgSizeY 95
//#define bandcount 50

std::vector<float> rescaleAffineMatrix(std::vector<float> v);
std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix);
//Image data is 16bit, unsigned short int
//Convert to float, 32bit
float* readImgFromFile(std::string filename);
torch::Tensor normalizeTensor(torch::Tensor tensor);   //Normalize tensor to 0~1
void saveAndCheckImage(float* imageData, int col_total, int row_total, int z_total, std::string name);   //Store images as tiff format
void saveAndCheckImage(unsigned short int* imageData, int col_total, int row_total, int z_total, std::string name);
void getFileNames(std::string path, std::vector<std::string>& files);   //Get the names of all files in a folder
std::string int2string(int n, int i);  //Number to character, specify length, e.g. 0001
std::string getTime();  //Get current time, YYYYMMDD_HHMM, eg: 20220711_0926

void generateMatFromYaml(cv::Mat& matrix, std::string dataName);
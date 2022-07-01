#pragma once
//libtorch
#include <torch/torch.h>
#include <torch/script.h>

#include <vector>
#include <string>



//crop��ͼ��Ĵ�С
//#define nImgSizeX 76
//#define nImgSizeY 95
//#define bandcount 50

std::vector<float> rescaleAffineMatrix(std::vector<float> v);
std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix);
//ͼ��������16bit�� unsigned short int
//ǿ��ת����float�� 32bit
float* readImgFromFile(std::string filename);
torch::Tensor normalizeTensor(torch::Tensor tensor);   //��tensor��һ��Ϊ0~1
void saveAndCheckImage(float* imageData, int col_total, int row_total, int z_total, std::string name);   //�洢ͼ��Ϊtiff��ʽ
void getFileNames(std::string path, std::vector<std::string>& files);   //��ȡ�ļ����������ļ���
std::string int2string(int n, int i);  //����ת�ַ���ָ�����ȣ���0001
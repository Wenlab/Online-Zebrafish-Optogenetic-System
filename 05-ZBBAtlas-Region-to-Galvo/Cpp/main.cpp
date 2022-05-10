#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include<io.h>
#include <sstream>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#define _USE_MATH_DEFINES
#include <math.h>



#include"zbb2FishImg.h"

#define  PI 3.1415926

using namespace std;
using namespace cv;

#pragma warning(disable : 4996)

// create a trackbar
const int max_width = 60;
const int max_height = 80;
int w_pos = 0;
int h_pos = 0;

std::vector<std::vector<float>> readParamsFromTXT(std::string filename, const char* split);
void getFileNames(std::string path, std::vector<std::string>& files);

void trackbar_width(int, void*)
{
	w_pos = getTrackbarPos("w_pos", "test_trackbar");
	return;
}
void trackbar_height(int, void*)
{
	h_pos = getTrackbarPos("w_pos", "test_trackbar");
	return;
}

int main()
{
	//测试zbb2FishImg
	zbb2FishImg FishReg;
	FishReg.initialize();

	namedWindow("test_trackbar", 1);
	//创建轨迹条
	createTrackbar("w_pos", "test_trackbar", &w_pos, max_width, trackbar_width);
	createTrackbar("h_pos", "test_trackbar", &h_pos, max_height, trackbar_height);


	//从txt文件中读取测试数据
	const char* ch1 = ",";
	const char* ch2 = " ";
	std::vector<std::vector<float>> crop = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/cropPoint.txt", ch1);
	std::vector<std::vector<float>> rotationAngleXY = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/rotationAngleXY.txt", ch1);
	std::vector<std::vector<float>> rotationAngleYZ = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/rotationAngleYZ.txt", ch1);
	std::vector<std::vector<float>> fix2movingParam = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/fix2movingParam.txt", ch2);

	//TEST IMAGES
	std::vector<std::string> imgfilenames;
	getFileNames("F:/ITK/matchingZBB/matchingZBB/TM_test5/MIP", imgfilenames);



	//cout << crop.size() << endl;
	//for (int j = 0; j < crop.size(); j++)
	//{
	//	for (int i = 0; i < crop[j].size(); i++)
	//	{
	//		cout << crop[j][i] << "    ";
	//	}
	//	cout << endl;
	//}

	vector<float> Fix2ZBBAM{ 0.985154,	0.0184487, - 0.00942914,
		-0.0166061,	1.13246, - 0.102937,
		0.0196408, - 0.0078765,	1.25844,
		0.522241, - 6.91866, - 11.7296 };
	//for(int i = 0; i< fix2movingParam.size(); i++)
	while (1)
	{
		for (int i = 0; i < 1; i++)
		{
			w_pos = getTrackbarPos("w_pos", "test_trackbar");
			h_pos = getTrackbarPos("h_pos", "test_trackbar");

			Rect lightArea(w_pos, h_pos, 10, 10);
			FishReg.getRegionFromUser(lightArea);


			//vector<float> Moving2FixAM{ 0.998911, -0.0885304, -0.00483924,
			//	0.121989,   0.988682,   0.0257309,
			//	0.000398744, -0.0268302,   1.01408,
			//	6.49591, -4.78014,   1.13219 };
			vector<float> Moving2FixAM = fix2movingParam[i];
			Point3f cropPoint(crop[i][1], crop[i][0], 0);   //matlab出的坐标和C是反的   如果全部整合成C 这里变成先0再1
			float AngleXY = rotationAngleXY[i][0];
			float AngleYZ = rotationAngleYZ[i][0];
			//Point3f cropPoint(54, 64, 0);
			//double rotationAngleXY = -153;
			////double rotationAngleZ = -160;
			//double rotationAngleYZ = 10;



			FishReg.getZBB2FixAffineMatrix(Fix2ZBBAM);
			FishReg.getFix2MovingAffineMatrix(Moving2FixAM);
			FishReg.getCropPoint(cropPoint);
			FishReg.getRotationMatrix(AngleXY, AngleYZ);

			FishReg.ZBB2FishTransform();   //已获取区域信息



			//test
			Mat zbbMIP = imread("Ref-zbb-MIP.png");
			rectangle(zbbMIP, lightArea, Scalar(255), 1);
			namedWindow("zbbMIP", 0);
			resizeWindow("zbbMIP", Size(zbbMIP.cols * 4, zbbMIP.rows * 4));
			imshow("zbbMIP", zbbMIP);

			//for (int i = 0; i < FishReg.BrainRegionName.size(); i++)
			//{
			//	cout << FishReg.BrainRegionName[i] << endl;
			//}

			vector<Point3f> RegionCoord_5 = FishReg.regionInFish;
			Mat origImg = imread(imgfilenames[i]);
			//origImg.setTo(0);
			for (int i = 0; i < RegionCoord_5.size(); i++)
			{
				Point3f p = RegionCoord_5[i];
				circle(origImg, Point(p.x, p.y), 1, Scalar(255));
			}
			RegionCoord_5.clear();
			imshow("result", origImg);


			//cout << i << endl;
			waitKey(30);


			FishReg.clear();
		}
	}

	return 0;
}


std::vector<std::vector<float>> readParamsFromTXT(std::string filename, const char* split)
{
	std::ifstream fin(filename);
	if (!fin)
	{
		cout << "error! can't open file" << endl;
	}

	std::vector<std::vector<float>> datas;
	std::vector<float> in;
	string s;

	while (getline(fin, s))
	{
		//string to char
		char *s_input = (char*)s.c_str();
		//const char * split = " ";
		char *buf;
		char *p = strtok_s(s_input, split, &buf);
		double a;
		while (p != NULL)
		{
			a = atof(p);
			in.push_back(a);
			p = strtok_s(NULL, split, &buf);
		}
		datas.push_back(in);
		in.clear();
	}
	fin.close();

	return datas;
}
 
void getFileNames(std::string path, std::vector<std::string>& files)
{
	//文件句柄
	//注意：我发现有些文章代码此处是long类型，实测运行中会报错访问异常
	intptr_t hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,递归查找
			//如果不是,把文件绝对路径存入vector中
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFileNames(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


//vector<pair<string, Point>> readZBBMapFromTxt(string file);
//
//vector<vector<vector<string>>> makeZbbMapVec(vector<pair<string, Point>> zbbMapPair);
//
//vector<string> queryRegionName(Rect region, vector<vector<vector<string>>> zbbMapVec);
//
//std::vector<float> inverseAffineMatrix(std::vector<float> am);
//
//Point3f applyAffineMatrixOn3DCoord(Point3f temp, vector<float> am);
//
////bool cmp(string a, string b)
////{
////	return strcmp(a.c_str(), b.c_str());
////}
//
//int main()
//{
//	string zbbPath = "anatomyList_4bin.txt";
//
//	vector<pair<string, Point>> zbbMapPair;
//	zbbMapPair = readZBBMapFromTxt(zbbPath);
//	vector<vector<vector<string>>> zbbMapVec = makeZbbMapVec(zbbMapPair);
//
//
//	//选择打光的位置
//	Rect lightArea(50, 50, 10, 10);
//	vector<string> regionName = queryRegionName(lightArea, zbbMapVec);
//	//check
//	//for (int i = 0; i < regionName.size(); i++)
//	//{
//	//	cout << regionName[i] << endl;
//	//}
//	vector<Point3f> RegionCoord;   //zbb上直接选的区域坐标是二维，为了3D对齐扩展到三维
//	for (int i = lightArea.x; i < lightArea.x + lightArea.width; i++)
//	{
//		for (int j = lightArea.y; j < lightArea.y + lightArea.height; j++)
//		{
//			Point3f temp{ (float)i,(float)j,0 };
//			RegionCoord.push_back(temp);
//			//cout << temp << endl;
//		}
//	}
//
//	//从zbb到fixImage的逆仿射变换
//	//初始化时候算的
//	vector<float> zbb2FixAM{ 0.995751 ,-0.0000857003 ,0.00319121 ,
//		-0.000161462 ,1.06928 ,0.000254662 ,
//		0.0136118 ,-0.0196541 ,1.22397 ,
//		0.0194115 ,0 ,0 };
//
//	vector<float> zbb2FixAM_inv = inverseAffineMatrix(zbb2FixAM);
//	vector<Point3f> RegionCoord_2;
//	for (int i = 0; i < RegionCoord.size(); i++)
//	{
//		Point3f temp = RegionCoord[i];
//		Point3f p = applyAffineMatrixOn3DCoord(temp, zbb2FixAM_inv);
//
//		RegionCoord_2.push_back(p);
//	}
//	//从fixImage到MovingImage的逆仿射变换
//	//从affine网络得到
//	vector<float> fix2MovingAM{ 0.993222 ,-0.10944 ,-0.0136222 ,
//								0.127151 ,0.987073 ,0.00369118 ,
//								0.0021671 ,-0.0144663 ,0.988299,
//								7.98918 ,-4.16694 ,0.343009 };
//
//	vector<Point3f> RegionCoord_3;
//	for (int i = 0; i < RegionCoord_2.size(); i++)
//	{
//		Point3f temp = RegionCoord_2[i];
//		Point3f p = applyAffineMatrixOn3DCoord(temp, zbb2FixAM_inv);
//
//		RegionCoord_3.push_back(p);
//	}
//
//	//从MovingImage(77*95*52)到原图(200*200*50)
//	//参数来自crop和模板匹配
//	Point3f cropPoint(53, 62, 0);
//	vector<Point3f> RegionCoord_4;
//	for (int i = 0; i < RegionCoord_3.size(); i++)
//	{
//		Point3f temp = RegionCoord_3[i];
//		Point3f p = temp + cropPoint;
//
//		RegionCoord_4.push_back(p);
//	}
//	//绕z轴的旋转矩阵
//	double rotationAngleZ = -130;
//	rotationAngleZ = rotationAngleZ * PI / 180.0;
//	Eigen::Matrix3f rotationMatrixZ;
//	rotationMatrixZ << cos(rotationAngleZ), -sin(rotationAngleZ), 0, 
//						sin(rotationAngleZ), cos(rotationAngleZ), 0, 
//						0, 0, 1;
//
//	cout << rotationMatrixZ << endl << endl;
//	//绕x轴的旋转
//	double rotationAngleX = 10;
//	rotationAngleX = rotationAngleX * PI / 180.0;
//	Eigen::Matrix3f rotationMatrixX;
//	rotationMatrixX << 1, 0, 0,
//		0, cos(rotationAngleX), -sin(rotationAngleX),
//		0, sin(rotationAngleX), cos(rotationAngleX);
//
//	cout << rotationMatrixX << endl << endl;
//
//	//两个旋转矩阵相乘求总旋转矩阵
//	Eigen::Matrix3f rotationMatrix = rotationMatrixZ;
//	cout << rotationMatrix << endl << endl;
//
//	//将旋转应用到坐标上
//	vector<Point3f> RegionCoord_5;
//	for (int i = 0; i < RegionCoord_4.size(); i++)
//	{
//		Point3f bias(100, 100, 25);
//		Point3f temp = RegionCoord_4[i];
//		temp = temp - bias;
//		Point3f p;
//		p.x = temp.x*rotationMatrix(0, 0) + temp.y*rotationMatrix(0, 1) + temp.z*rotationMatrix(0, 2);
//		p.y = temp.x*rotationMatrix(1, 0) + temp.y*rotationMatrix(1, 1) + temp.z*rotationMatrix(1, 2);
//		p.z = temp.x*rotationMatrix(2, 0) + temp.y*rotationMatrix(2, 1) + temp.z*rotationMatrix(2, 2);
//		p = p + bias;
//		cout << p << endl;
//		RegionCoord_5.push_back(p);
//		cout << p << endl;
//	}
//
//
//
//	Mat zbbMIP = imread("Ref-zbb-MIP.png");
//	rectangle(zbbMIP, lightArea, Scalar(255), 1);
//
//	namedWindow("zbbMIP", 0);
//	resizeWindow("zbbMIP", Size(zbbMIP.cols * 4, zbbMIP.rows * 4));
//
//	imshow("zbbMIP", zbbMIP);
//
//
//	Mat origImg(Size(200, 200), CV_8UC3);
//	origImg.setTo(0);
//	for (int i = 0; i < RegionCoord_5.size(); i++)
//	{
//		Point3f p = RegionCoord_5[i];
//		circle(origImg, Point(p.x, p.y), 1, Scalar(255));
//	}
//
//	imshow("result", origImg);
//
//	waitKey(0);
//
//	return 0;
//}
//
//
//vector<pair<string, Point>> readZBBMapFromTxt(string file)
//{
//	vector<pair<string, Point>> zbbMap;
//
//	ifstream infile;
//	infile.open(file.data());   //将文件流对象与文件连接起来 
//	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
//
//	string str, pattern;
//	pattern = ",";
//	while (getline(infile, str))
//	{
//		char * strc = new char[strlen(str.c_str()) + 1];
//		strcpy(strc, str.c_str());
//		vector<string> res;
//		char* temp = strtok(strc, pattern.c_str());
//		while (temp != NULL)
//		{
//			res.push_back(string(temp));
//			temp = strtok(NULL, pattern.c_str());
//		}
//		delete[] strc;
//
//		pair<string, Point> t{ res[0],Point(atof(res[1].c_str()),atof(res[2].c_str())) };
//		zbbMap.push_back(t);
//		res.clear();
//		//cout << str << endl;
//	}
//	infile.close();             //关闭文件输入流 
//
//	return zbbMap;
//}
//
//vector<vector<vector<string>>> makeZbbMapVec(vector<pair<string, Point>> zbbMapPair)
//{
//	int a = 77;
//	int b = 95;
//	vector<vector<vector<string>>> zbbMapVec(a, vector<vector<string>>(b));
//
//	//cout << zbbMapPair.size() << endl;
//	for (int i = 0; i < zbbMapPair.size(); i++)
//	{
//		pair<string, Point> temp = zbbMapPair[i];
//		zbbMapVec[temp.second.x - 1][temp.second.y - 1].push_back(temp.first);   //matlab保存的坐标是从1开始
//	}
//	//去除每个像素位置的相同元素
//	for (int i = 0; i < zbbMapVec.size(); i++)
//	{
//		for (int j = 0; j < zbbMapVec[i].size(); j++)
//		{
//			zbbMapVec[i][j].erase(unique(zbbMapVec[i][j].begin(), zbbMapVec[i][j].end()), zbbMapVec[i][j].end());
//			//cout << zbbMapVec[i][j].size() << endl;
//		}
//	}
//	return zbbMapVec;
//}
//
//vector<string> queryRegionName(Rect region, vector<vector<vector<string>>> zbbMapVec)
//{
//	vector<string> queryRegion;
//	for (int i = region.x; i < region.x + region.width; i++)
//	{
//		for (int j = region.y; j < region.y + region.height; j++)
//		{
//			vector<string> pixelLable = zbbMapVec[i][j];
//			//cout << pixelLable.size() << endl;
//			for (int m = 0; m < pixelLable.size(); m++)
//			{
//				queryRegion.push_back(pixelLable[m]);
//			}
//		}
//	}
//
//	//删除重复的区域名称
//	sort(queryRegion.begin(), queryRegion.end());
//	queryRegion.erase(unique(queryRegion.begin(), queryRegion.end()), queryRegion.end());
//
//	return queryRegion;
//}
//
//std::vector<float> inverseAffineMatrix(std::vector<float> am)
//{
//
//	Eigen::Matrix3f am_eigen;  //不包含三个平移参数
//	for (int a = 0; a < 3; a++)
//	{
//		for (int b = 0; b < 3; b++)
//		{
//			am_eigen(b, a) = am[a * 3 + b];
//		}
//	}
//	Eigen::Matrix3f am_eigen_inverse = am_eigen.inverse();   //求出的逆和原矩阵相乘结果不为1？
//	//std::cout << am_eigen_reverse * am_eigen << std::endl;
//
//	Eigen::Vector3f am_trans;
//	am_trans << am[9], am[10], am[11];
//	Eigen::Vector3f am_trans_inverse = am_trans.transpose() * (-am_eigen_inverse);
//
//
//	//std::cout << am_trans << std::endl <<std::endl<< am_trans_inverse << std::endl;
//	std::vector<float> am_inverse(12);
//	for (int a = 0; a < 3; a++)
//	{
//		for (int b = 0; b < 3; b++)
//		{
//			am_inverse[a * 3 + b] = am_eigen_inverse(b, a);
//		}
//	}
//
//	am_inverse[9] = am_trans_inverse(0);
//	am_inverse[10] = am_trans_inverse(1);
//	am_inverse[11] = am_trans_inverse(2);
//
//	return am_inverse;
//}
//
//Point3f applyAffineMatrixOn3DCoord(Point3f temp, vector<float> am)
//{
//	Point3f p;
//	p.x = temp.x*am[0] + temp.y*am[1] + temp.z*am[2] + am[9];
//	p.y = temp.x*am[3] + temp.y*am[4] + temp.z*am[5] + am[10];
//	p.z = temp.x*am[6] + temp.y*am[7] + temp.z*am[8] + am[11];
//
//	return p;
//}
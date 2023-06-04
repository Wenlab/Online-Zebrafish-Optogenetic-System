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
	//test zbb2FishImg
	zbb2FishImg FishReg;
	FishReg.initialize();

	namedWindow("test_trackbar", 1);
	//Create trackbar
	createTrackbar("w_pos", "test_trackbar", &w_pos, max_width, trackbar_width);
	createTrackbar("h_pos", "test_trackbar", &h_pos, max_height, trackbar_height);


	//Read test data from txt file
	const char* ch1 = ",";
	const char* ch2 = " ";
	std::vector<std::vector<float>> crop = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/cropPoint.txt", ch1);
	std::vector<std::vector<float>> rotationAngleXY = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/rotationAngleXY.txt", ch1);
	std::vector<std::vector<float>> rotationAngleYZ = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/rotationAngleYZ.txt", ch1);
	std::vector<std::vector<float>> fix2movingParam = readParamsFromTXT("F:/ITK/matchingZBB/matchingZBB/TM_test5/params/fix2movingParam.txt", ch2);

	//TEST IMAGES
	std::vector<std::string> imgfilenames;
	getFileNames("F:/ITK/matchingZBB/matchingZBB/TM_test5/MIP", imgfilenames);


	vector<float> Fix2ZBBAM{ 0.985154,	0.0184487, - 0.00942914,
		-0.0166061,	1.13246, - 0.102937,
		0.0196408, - 0.0078765,	1.25844,
		0.522241, - 6.91866, - 11.7296 };

	while (1)
	{
		for (int i = 0; i < 1; i++)
		{
			w_pos = getTrackbarPos("w_pos", "test_trackbar");
			h_pos = getTrackbarPos("h_pos", "test_trackbar");

			Rect lightArea(w_pos, h_pos, 10, 10);
			FishReg.getRegionFromUser(lightArea);

			vector<float> Moving2FixAM = fix2movingParam[i];
			Point3f cropPoint(crop[i][1], crop[i][0], 0);   //The coordinates output from matlab and C are inverted
			float AngleXY = rotationAngleXY[i][0];
			float AngleYZ = rotationAngleYZ[i][0];



			FishReg.getZBB2FixAffineMatrix(Fix2ZBBAM);
			FishReg.getFix2MovingAffineMatrix(Moving2FixAM);
			FishReg.getCropPoint(cropPoint);
			FishReg.getRotationMatrix(AngleXY, AngleYZ);

			FishReg.ZBB2FishTransform();   //Acquired area information



			//test
			Mat zbbMIP = imread("Ref-zbb-MIP.png");
			rectangle(zbbMIP, lightArea, Scalar(255), 1);
			namedWindow("zbbMIP", 0);
			resizeWindow("zbbMIP", Size(zbbMIP.cols * 4, zbbMIP.rows * 4));
			imshow("zbbMIP", zbbMIP);


			vector<Point3f> RegionCoord_5 = FishReg.regionInFish;
			Mat origImg = imread(imgfilenames[i]);
			for (int i = 0; i < RegionCoord_5.size(); i++)
			{
				Point3f p = RegionCoord_5[i];
				circle(origImg, Point(p.x, p.y), 1, Scalar(255));
			}
			RegionCoord_5.clear();
			imshow("result", origImg);

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
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
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



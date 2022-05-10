#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#define _USE_MATH_DEFINES
#include <math.h>

#define  PI 3.1415926


#pragma once
class zbb2FishImg
{

public:
	zbb2FishImg();
	~zbb2FishImg();

	cv::Rect Region;  
	std::vector<std::string> BrainRegionName;
	std::vector<cv::Point3f> RegionCoord;

	std::vector<std::vector<std::vector<std::string>>> zbbMapVec;
	std::vector<float> zbb2FixAM;
	std::vector<float> Fix2MovingAM;
	cv::Point3f cropPoint;
	Eigen::Matrix3f rotationMatrix;
	std::vector<cv::Point3f> regionInFish;   //to galvo

	void initialize();
	void getRegionFromUser(cv::Rect Reg);
	void getZBB2FixAffineMatrix(std::vector<float> Fix2zbbAM);
	void getFix2MovingAffineMatrix(std::vector<float> Moving2FixAM);
	void getCropPoint(cv::Point3f pt);
	void getRotationMatrix(float rotationAngleZ, float rotationAngleX);

	void ZBB2FishTransform();
	std::vector<std::pair<std::string, cv::Point>> readZBBMapFromTxt(std::string file);
	std::vector<std::vector<std::vector<std::string>>> makeZbbMapVec(std::vector<std::pair<std::string, cv::Point>> zbbMapPair);
	std::vector<std::string> queryRegionName(cv::Rect region);
	void clear();
};

std::vector<float> inverseAffineMatrix(std::vector<float> am);
cv::Point3f applyAffineMatrixOn3DCoord(cv::Point3f temp, std::vector<float> am);
cv::Point3f applyRotationMatrixOn3DCoord(cv::Point3f temp, Eigen::Matrix3f rotationMatrix);
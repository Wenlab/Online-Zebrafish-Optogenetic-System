#pragma once
#include"imgProcess.h"

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>



struct OptogeneticParams
{
	int xsize = 10;
	int xpos = 10;
	int ysize = 10;
	int ypos = 10;
	int laserTime = 2000;
	int laserOn = 0;
};

class Experiment
{
public:
	int frameNum;

	unsigned short int *Image;
	unsigned short int *Image4bin;


	OptogeneticParams params;
	int recordOn;
	int UserWantToStop;

	cv::Rect roi;//用户在ref上选的给光区域
	std::vector<cv::Point3f> ROIpoints;//坐标转换到鱼身上的点

	cv::Mat MIP;
	cv::Mat ref;
	cv::Mat ref_resize;
	cv::Mat ref_MIP;

	Experiment(std::string modle_path);
	~Experiment();


	FishImageProcess fishImgProc;

	//write out
	std::string folderName;
	std::string txtName;
	std::ofstream outTXT;

	void initialize();
	void resizeImg();
	void ImgReconAndRegis();
	void saveImg2Disk(std::string filename);

	void getReconMIP();

	void setupGUI();
	void getParamsFromGUI();
	void drawGUIimg();

	void initializeWriteOut();
	void writeOutTxt();
	
	void controlExp();


	void clear();
	void exit();

	//线程函数
	void writeOutData();
	void imgProcess();
	void galvoControl();
	void readFullSizeImgFromFile();
	void readFullSizeImgFromCamera();

private:

};



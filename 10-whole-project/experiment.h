#pragma once
#include"imgProcess.h"

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>


class Experiment
{
public:

	unsigned short int *Image;
	unsigned short int *Image4bin;


	cv::Mat MIP;

	Experiment(std::string modle_path);
	~Experiment();


	FishImageProcess fishImgProc;

	void prepareMemory();
	void initialize();
	void readFullSizeImgFromFile(std::string filename);
	void readFullSizeImgFromCamera();
	void resizeImg();
	void ImgReconAndRegis();
	void saveImg2Disk(std::string filename);

	void getReconMIP();


private:

};



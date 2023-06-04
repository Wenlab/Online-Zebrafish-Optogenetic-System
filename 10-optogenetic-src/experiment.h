#pragma once

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include"imgProcess.h"
#include"Talk2Camera.h"
#include"Talk2Galvo.h"
#include"TCPserver.h"





struct OptogeneticParams
{
	int xsize = 10; //Size of optogenetic region
	int xpos = 10;  //Location of optogenetic region
	int ysize = 10;
	int ypos = 10;
	int xbias = 5;  //Bias per fish fine tuned
	int ybias = 5;
	int laserTime = 1500;
	int laserOn = 0;
	int laserOnLongTime = 0;
	int interval = 30;   // stimulation interval in seconds
	int circleNum = 30;  //Number of stimulation cycles
	int circleOn = 0;   //Whether cyclic stimulation

};

class Experiment
{
public:
	int frameNum;

	unsigned short int *Image;
	unsigned short int *Image4bin;
	float* mip_cpu;
	float* cropResult_cpu;

	cv::Mat ref_mip_resize;
	cv::VideoWriter MIPWriter;

	//cameras
	AT_H cam_handle; 
	CamData* cam_data;



	OptogeneticParams params;
	int circleCount;
	int recordOn;
	int UserWantToStop;

	cv::Rect roi;//The stimulus area selected by the user on ZBB
	std::vector<cv::Point2f> ROIpoints;//Result of coordinate conversion

	cv::Mat MIP;

	int maxValue;
	int thre;


	FishImageProcess fishImgProc;

	//write out
	std::string rawfolderName;
	std::string cropfolderName;
	std::string txtName;
	std::ofstream outTXT;
	std::ofstream timeTXT;

	//galvo control
	double galvoXmin;
	double galvoYmin;
	GalvoData galvo;
	cv::Mat galvoMatrixX;
	cv::Mat galvoMatrixY;
	double scale;
	std::vector<cv::Point2f> galvoVotagesPairs;

	//Detecting if fish are moving
	std::vector<int> headingAngleVec;
	bool fishMoving;
	bool movingPause;


	//TCP server
	TCPServer server;

	Experiment(std::string modle_path);
	~Experiment();

	void initialize();
	void resizeImg();
	void ImgReconAndRegis();
	//void saveImg2Disk(std::string filename);

	void getReconMIP();

	void setupGUI();
	void getParamsFromGUI();
	void drawGUIimg();

	void initializeWriteOut();
	void writeOutTxt();

	void controlExp();

	void generateGalvoVotages();


	void preProcessImg();

	void isFishMoving(int angle);

	void clear();
	void exit();

	//Thread Functions
	void writeOutData();
	void imgProcess();
	void galvoControl();
	void TCPconnect();
	void generateGalvoVoltagesPairs();
	void readFullSizeImgFromFile();
	void readFullSizeImgFromCamera();



private:

};
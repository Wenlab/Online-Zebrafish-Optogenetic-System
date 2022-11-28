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
	int xsize = 10; //光遗传选区的大小
	int xpos = 10;  //光遗传选区的位置
	int ysize = 10;
	int ypos = 10;
	int xbias = 5;  //按照threshold，选取鱼的包围矩形，包围矩形左上点与选取点的bias
	int ybias = 5;
	int laserTime = 1500;
	int laserOn = 0;
	int laserOnLongTime = 0;
	int interval = 30;   // 给光间隔  单位是s
	int circleNum = 30;  //给多少个周期
	int circleOn = 0;   //是否周期性给光

};

class Experiment
{
public:
	int frameNum;

	unsigned short int *Image;
	unsigned short int *Image_forSave;
	unsigned short int *Image4bin;
	float* mip_cpu;
	float* cropResult_cpu;

	cv::Mat ref_mip_resize;
	cv::VideoWriter MIPWriter;

	//cameras
	AT_H cam_handle; //相机句柄变量的声明,随后将再初始化时为其赋值,之后用它来传递相机信息
	CamData* cam_data;



	OptogeneticParams params;
	int circleCount;
	int recordOn;
	int UserWantToStop;

	cv::Rect roi;//用户在ref上选的给光区域
	std::vector<cv::Point2f> ROIpoints;//坐标转换到鱼身上的点

	cv::Mat MIP;

	//相机图像读进来，截断亮度700以上的像素
	int maxValue;
	int thre;


	FishImageProcess fishImgProc;

	//write out
	std::string rawfolderName;
	std::string cropfolderName;
	std::string txtName;
	std::ofstream outTXT;

	//galvo control
	double galvoXmin;
	double galvoYmin;
	GalvoData galvo;
	cv::Mat galvoMatrixX;
	cv::Mat galvoMatrixY;
	double scale;
	std::vector<cv::Point2f> galvoVotagesPairs;

	//检测鱼是否在运动，运动时关闭galvo
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

	//检测鱼是否在运动
	void isFishMoving(int angle);

	void clear();
	void exit();

	//线程函数
	void writeOutData();
	void imgProcess();
	void galvoControl();
	void TCPconnect();
	void generateGalvoVoltagesPairs();
	void readFullSizeImgFromFile();
	void readFullSizeImgFromCamera();



private:

};
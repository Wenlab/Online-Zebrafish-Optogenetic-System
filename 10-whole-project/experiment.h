#pragma once
#include"imgProcess.h"
#include"Talk2Camera.h"
#include"Talk2Galvo.h"

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
	unsigned short int *Image_forSave;
	unsigned short int *Image4bin;
	float* mip_cpu;

	//cameras
	AT_H cam_handle; //相机句柄变量的声明,随后将再初始化时为其赋值,之后用它来传递相机信息
	CamData* cam_data;



	OptogeneticParams params;
	int recordOn;
	int UserWantToStop;

	cv::Rect roi;//用户在ref上选的给光区域
	std::vector<cv::Point2f> ROIpoints;//坐标转换到鱼身上的点

	cv::Mat MIP;

	Experiment(std::string modle_path);
	~Experiment();


	FishImageProcess fishImgProc;

	//write out
	std::string folderName;
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

	void generateGalvoVotages();


	void clear();
	void exit();

	//线程函数
	void writeOutData();
	void imgProcess();
	void galvoControl();
	void readFullSizeImgFromFile();
	void readFullSizeImgFromCamera();

	void flipImage();

private:

};



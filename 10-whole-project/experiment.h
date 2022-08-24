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
	int xsize = 10; //���Ŵ�ѡ���Ĵ�С
	int xpos = 10;  //���Ŵ�ѡ����λ��
	int ysize = 10;
	int ypos = 10;
	int xbias = 5;  //����threshold��ѡȡ��İ�Χ���Σ���Χ�������ϵ���ѡȡ���bias
	int ybias = 5;
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
	float* cropResult_cpu;

	//cameras
	AT_H cam_handle; //����������������,����ٳ�ʼ��ʱΪ�丳ֵ,֮�����������������Ϣ
	CamData* cam_data;



	OptogeneticParams params;
	int recordOn;
	int UserWantToStop;

	cv::Rect roi;//�û���ref��ѡ�ĸ�������
	std::vector<cv::Point2f> ROIpoints;//����ת���������ϵĵ�

	cv::Mat MIP;

	//���ͼ����������ض�����700���ϵ�����
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

	void clear();
	void exit();

	//�̺߳���
	void writeOutData();
	void imgProcess();
	void galvoControl();
	void TCPconnect();
	void generateGalvoVoltagesPairs();
	void readFullSizeImgFromFile();
	void readFullSizeImgFromCamera();



private:

};
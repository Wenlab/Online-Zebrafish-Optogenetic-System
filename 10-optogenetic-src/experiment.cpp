#include"experiment.h"
#include"kexinLibs.h"
#include "gdal_priv.h"
#include <gdal.h>

#include"Timer.h"

#include  <io.h>  
#include  <stdio.h>  
#include  <stdlib.h> 

#include"avir.h"

#include<mutex>

#include <opencv2/core/utils/logger.hpp>

#pragma warning(disable:4996)

using namespace std;


struct image_time_pair
{
	unsigned short int* imagedata;
	char time_stamp_s[256];
	int frameNum = 0;
};

queue<image_time_pair> SaveImageTimeQueue;


Experiment::Experiment(std::string modle_path) :fishImgProc(modle_path)
{
	maxValue = 65535;
	thre = 1000;
}

Experiment::~Experiment()
{

}


void Experiment::initialize()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT); 

	Image = new unsigned short int[2048 * 2048 * 1]; 
	Image4bin = new unsigned short int[512 * 512 * 1];
	mip_cpu = new float[200 * 200 * 1];
	cropResult_cpu = new float[76 * 95 * 50]();

	MIP = cv::Mat(200, 200, CV_32FC1, mip_cpu);

	////Initializing the camera
	cam_data = T2Cam_CreateCamData();
	T2Cam_InitializeLib(&cam_handle);
	SetupBinningandAOI(cam_handle);
	T2Cam_InitializeCamData(cam_data, cam_handle);
	getUserSettings(cam_handle);
	CreateBuffer(cam_data, cam_handle);
	cout << "camera prepare done" << endl;

	////Initialize galvo
	galvoXmin = -10;
	galvoYmin = -10;
	galvo.initialize();
	generateMatFromYaml(galvoMatrixX, "GalvoX_matrix");
	generateMatFromYaml(galvoMatrixY, "GalvoY_matrix");
	minMaxLoc(galvoMatrixX, &galvoXmin);
	minMaxLoc(galvoMatrixY, &galvoYmin);

	fishImgProc.initialize();
	cout << "initialize fishImgProc done" << endl;

	initializeWriteOut();
	cout << "write out prepare done" << endl;
	setupGUI();
	cout << "GUI prepare done" << endl;



	frameNum = 0;
	recordOn = 0;
	circleCount = 0;
	UserWantToStop = 0;

	fishMoving = false;
	movingPause = false;

	return;
}

void Experiment::readFullSizeImgFromFile()
{
	cout << "read Image thread say hello!!" << endl;

	string beforeResizeImgPath = "D:/kexin/Online-Zebrafish-Optogenetic/data/r20210824_X10";
	vector<string> beforeResizeImgNames;
	getFileNames(beforeResizeImgPath, beforeResizeImgNames);

	SYSTEMTIME currentTime;
	char time_stamp_s[256] = { 0 };

	while (!UserWantToStop)
	{
		string filename = beforeResizeImgNames[frameNum];
		GDALDataset *poDataset;  
		poDataset = (GDALDataset *)GDALOpen(filename.data(), GA_ReadOnly);
		if (poDataset == NULL)
		{
			cout << "fail in open files!!!" << endl;
			continue;
		}
		int nImgSizeX = poDataset->GetRasterXSize();
		int nImgSizeY = poDataset->GetRasterYSize();
		GDALRasterBand *poBand;
		int bandcount = poDataset->GetRasterCount();	
		if (DEBUG)
		{
			cout << nImgSizeX << "  *  " << nImgSizeY << endl;
			cout << "band num£º" << bandcount << endl;
		}
		int num_iamge_size = 0;
		for (int bandind = 1; bandind <= bandcount; bandind++)
		{
			poBand = poDataset->GetRasterBand(bandind);
			unsigned short int *pafScanline = new unsigned short int[nImgSizeX*nImgSizeY];
			poBand->RasterIO(GF_Read, 0, 0, nImgSizeX, nImgSizeY, pafScanline, nImgSizeX, nImgSizeY, GDALDataType(poBand->GetRasterDataType()), 0, 0);
			for (int i = 0; i < nImgSizeX; i++)
			{
				for (int j = 0; j < nImgSizeY; j++)
				{
					num_iamge_size++;
					Image[(bandind - 1) * nImgSizeX * nImgSizeY + i * nImgSizeY + j] = unsigned short int(pafScanline[i*nImgSizeY + j]);
				}
			}
			delete[] pafScanline;
		}


		resizeImg();
		if (recordOn)
		{
			unsigned short int* Image_forSave = new unsigned short int[2048 * 2048 * 1];
			memcpy(Image_forSave, Image, CCDSIZEX * CCDSIZEY * sizeof(unsigned short));

			GetLocalTime(&currentTime);
			sprintf(time_stamp_s, "%d_%d_%d_%d_%d", currentTime.wDay, currentTime.wHour, currentTime.wMinute, currentTime.wSecond, currentTime.wMilliseconds);

			image_time_pair image_time_pair_temp;
			image_time_pair_temp.imagedata = Image_forSave;
			strcpy(image_time_pair_temp.time_stamp_s, time_stamp_s);
			image_time_pair_temp.frameNum = frameNum;
			SaveImageTimeQueue.push(image_time_pair_temp);
		}

		frameNum = frameNum + 1;

		if (frameNum >= beforeResizeImgNames.size() - 1)
		{
			frameNum = 0;
		}
	}

	cout << "read Image thread say goodbye!!" << endl;


	return;
}


void Experiment::readFullSizeImgFromCamera()
{

	SYSTEMTIME currentTime;
	char time_stamp_s[256] = { 0 };

	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle);
	while (!UserWantToStop)
	{
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0)
		{
			GetLocalTime(&currentTime);
			sprintf(time_stamp_s, "%d_%d_%d_%d_%d", currentTime.wDay, currentTime.wHour, currentTime.wMinute, currentTime.wSecond, currentTime.wMilliseconds);
			memcpy(Image, cam_data->ImageRawData, CCDSIZEX * CCDSIZEY * sizeof(unsigned short));
		}
		else
		{
			cout << "Error in imaging acquisition!" << endl;
			break;
		}

		preProcessImg();
		resizeImg();


		if (recordOn)
		{
			unsigned short int* Image_forSave = new unsigned short int[2048 * 2048 * 1];
			memcpy(Image_forSave, Image, CCDSIZEX * CCDSIZEY * sizeof(unsigned short));


			image_time_pair image_time_pair_temp;
			image_time_pair_temp.imagedata = Image_forSave;
			strcpy(image_time_pair_temp.time_stamp_s, time_stamp_s);
			image_time_pair_temp.frameNum = frameNum;
			SaveImageTimeQueue.push(image_time_pair_temp);
		}
		frameNum = frameNum + 1;
	}

	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;

	return;
}

void Experiment::preProcessImg()
{
	cv::Mat temp(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1, Image);
	cv::flip(temp, temp, 0);   //flipcode=0, Vertical Flip Image

	return;
}


void Experiment::resizeImg()
{
	avir::CImageResizer<> ImageResizer(16);

	ImageResizer.resizeImage(Image, 2048, 2048, 0, Image4bin, 512, 512, 1, 0);

	return;
}



void Experiment::ImgReconAndRegis()
{
	fishImgProc.loadImage(Image4bin);
	/////Reconstruction
	fishImgProc.reconImage();
	fishImgProc.cropReconImage();
	////rotation
	fishImgProc.matchingANDrotationXY();
	////crop
	fishImgProc.cropRotatedImage(params.xbias, params.ybias);
	////affine
	fishImgProc.libtorchModelProcess();
	////Coordinate conversion
	ROIpoints = fishImgProc.ZBB2FishTransform(roi);

	generateGalvoVotages();


	return;
}



void Experiment::getReconMIP()
{
	cudaMemcpy(mip_cpu, fishImgProc.image2D_XY_gpu, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);
	
	cv::threshold(MIP, MIP, 10000, maxValue, CV_THRESH_TRUNC);

	cv::normalize(MIP, MIP, 0, 1, cv::NormTypes::NORM_MINMAX);

	return;
}

void Experiment::setupGUI()
{
	//set up GUI
	cout << "creating control panel for optogenetic..." << endl;
	cv::namedWindow("Control Panel for Optogenetic", cv::WINDOW_NORMAL);
	cv::resizeWindow("Control Panel for Optogenetic", 700, 700);

	cv::createTrackbar("recordOn", "Control Panel for Optogenetic", &recordOn, 1);
	cv::createTrackbar("LaserLong", "Control Panel for Optogenetic", &params.laserOnLongTime, 1);
	cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
	cv::createTrackbar("Interval", "Control Panel for Optogenetic", &params.interval, 60);  
	cv::createTrackbar("ScanTime", "Control Panel for Optogenetic", &params.laserTime, 5000); 
	cv::createTrackbar("circleNum", "Control Panel for Optogenetic", &params.circleNum, 100);  
	cv::createTrackbar("circleOn", "Control Panel for Optogenetic", &params.circleOn, 1);  

	cv::createTrackbar("xsize", "Control Panel for Optogenetic", &params.xsize, 50);
	cv::createTrackbar("ysize", "Control Panel for Optogenetic", &params.ysize, 50);
	cv::createTrackbar("xpos", "Control Panel for Optogenetic", &params.xpos, 76);
	cv::createTrackbar("ypos", "Control Panel for Optogenetic", &params.ypos, 95);
	cv::createTrackbar("xbias", "Control Panel for Optogenetic", &params.xbias, 20);
	cv::createTrackbar("ybias", "Control Panel for Optogenetic", &params.ybias, 20);

	cv::createTrackbar("Stop", "Control Panel for Optogenetic", &UserWantToStop, 1);

	cv::waitKey(1);
	return;
}


void Experiment::getParamsFromGUI()
{
	//get params from GUI
	recordOn = cv::getTrackbarPos("recordOn", "Control Panel for Optogenetic");
	params.laserOnLongTime = cv::getTrackbarPos("LaserLong", "Control Panel for Optogenetic");
	params.laserOn = cv::getTrackbarPos("LaserOn", "Control Panel for Optogenetic");
	params.interval=cv::getTrackbarPos("Interval", "Control Panel for Optogenetic");
	params.laserTime = cv::getTrackbarPos("ScanTime", "Control Panel for Optogenetic");
	params.circleNum=cv::getTrackbarPos("circleNum", "Control Panel for Optogenetic");
	params.circleOn=cv::getTrackbarPos("circleOn", "Control Panel for Optogenetic");

	params.xsize = cv::getTrackbarPos("xsize", "Control Panel for Optogenetic");
	params.ysize = cv::getTrackbarPos("ysize", "Control Panel for Optogenetic");
	params.xpos = cv::getTrackbarPos("xpos", "Control Panel for Optogenetic");
	params.ypos = cv::getTrackbarPos("ypos", "Control Panel for Optogenetic");
	params.xbias = cv::getTrackbarPos("xbias", "Control Panel for Optogenetic");
	params.ybias = cv::getTrackbarPos("ybias", "Control Panel for Optogenetic");

	UserWantToStop = cv::getTrackbarPos("Stop", "Control Panel for Optogenetic");

	if ((params.xpos + params.xsize) > 76)
		params.xsize = 76 - params.xpos;
	if ((params.ypos + params.ysize) > 95)
		params.ysize = 95 - params.ypos;
	roi = cv::Rect(params.xpos, params.ypos, params.xsize, params.ysize);

	cv::waitKey(1);
	return;
}


void Experiment::drawGUIimg()
{

	cv::Mat ref_MIP(200, 400, CV_8UC1, cv::Scalar(0));
	cv::Mat ref_resize;

	//Prepare the reference image, ZBB
	cv::Mat ref = cv::imread("Ref-zbb-MIP.png", 0);   //0£ºread gray image

	cv::rectangle(ref, roi, cv::Scalar(255), 1);
	cv::resize(ref, ref_resize, cv::Size(ref.cols * 2, ref.rows * 2));
	cv::Mat refROI = ref_MIP(cv::Rect(0, 0, ref_resize.cols, ref_resize.rows));
	ref_resize.copyTo(refROI);

	//Prepare the MIP for the current frame
	cv::Mat MIP_8u = MIP.clone();
	MIP_8u = (MIP_8u * 255);
	MIP_8u.convertTo(MIP_8u, CV_8UC1);
	//cout << "ROIpoints: " << ROIpoints.size() << endl;
	if (ROIpoints.size() > 3)
	{
		cv::RotatedRect re = cv::minAreaRect(ROIpoints);
		//Get the four vertices of the rotated rectangle
		cv::Point2f* vertices = new cv::Point2f[4];
		re.points(vertices);

		//Edge-by-edge drawing
		for (int j = 0; j < 4; j++)
		{
			cv::line(MIP_8u, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255), 1);
		}
	}



	cv::Mat MIPROI = ref_MIP(cv::Rect(200, 0, MIP_8u.cols, MIP_8u.rows));
	MIP_8u.copyTo(MIPROI);


	//cv::Mat ref_mip_resize;
	cv::resize(ref_MIP, ref_mip_resize, cv::Size(ref_MIP.cols * 2, ref_MIP.rows * 2));

	cv::imshow("test", ref_mip_resize);


	cv::waitKey(1);

	return;
}


void Experiment::initializeWriteOut()
{
	GDALAllRegister(); OGRRegisterAll();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");

	string path = "E:/online-opto-data/";
	// change to you own data path
	string time = getTime();
	string fishtype;
	string fishage;
	cout << "please input fish type" << endl;
	cin >> fishtype;
	cout << "please input fish age (e.g: 9)" << endl;
	cin >> fishage;

	string fishName = time + "_" + fishtype + "_" + fishage + "dpf";
	string folderName = path + fishName + "/";
	int ret = _access(folderName.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(folderName.c_str());
	}

	MIPWriter.open(folderName+"referenceMIP.avi", CV_FOURCC('M', 'J', 'P', 'G'),
		10, 
		cv::Size(800, 400),
		false);


	rawfolderName = folderName + "/raw/";
	cropfolderName = folderName + "/crop/";
	ret = _access(rawfolderName.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(rawfolderName.c_str());
	}
	ret = _access(cropfolderName.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(cropfolderName.c_str());
	}

	txtName = path + fishName + "/" + fishName + ".txt";
	outTXT.open(txtName);

	string timeStampTxtName = path + fishName + "/" + "timeStamp.txt";
	timeTXT.open(timeStampTxtName);

	return;
}

void Experiment::writeOutTxt()
{
	outTXT << "frameNum:" << frameNum << endl;
	if (params.laserOnLongTime)
	{
		outTXT << "laserOn:" << params.laserOnLongTime << endl;
	}
	else
	{
		outTXT << "laserOn:" << params.laserOn << endl;
	}
	outTXT << "fishMoving:" << fishMoving << endl;
	outTXT << "xsize:" << params.xsize << endl;
	outTXT << "xpos:" << params.xpos << endl;
	outTXT << "ysize:" << params.ysize << endl;
	outTXT << "ypos:" << params.ypos << endl;
	outTXT << "rotationAngleX:" << fishImgProc.rotationAngleX << endl;
	outTXT << "rotationAngleY:" << fishImgProc.rotationAngleY << endl;
	outTXT << "cropPoint:" << fishImgProc.cropPoint << endl;
	outTXT << "Moving2FixAffineMatrix:";
	outTXT << "circleCount:" << circleCount << endl; 

	for (int i = 0; i < fishImgProc.Moving2FixAM.size(); i++)
	{
		outTXT << fishImgProc.Moving2FixAM[i] << "    ";
	}
	outTXT << endl << endl;
}


void Experiment::writeOutData()
{
	cout << "write out thread say hello :p" << endl;


	int frameCount_writeOut = 0;
	while (!UserWantToStop)
	{
		if (SaveImageTimeQueue.size()>1)
		{
			image_time_pair savePair = SaveImageTimeQueue.front();
			saveAndCheckImage(savePair.imagedata, CCDSIZEX, CCDSIZEY, 1, rawfolderName + "/" + int2string(6, savePair.frameNum) + ".tif");
			timeTXT << savePair.time_stamp_s << endl;
			SaveImageTimeQueue.pop();
			delete savePair.imagedata;
			savePair.imagedata = NULL;
		}

		if (recordOn&&frameCount_writeOut != frameNum)
		{
			cudaMemcpy(cropResult_cpu, fishImgProc.ObjCropRed_gpu, sizeof(float) * 76 * 95 * 50, cudaMemcpyDeviceToHost);   //1
			saveAndCheckImage(cropResult_cpu, 76, 95, 50, cropfolderName + "/" + int2string(6, frameNum) + ".tif");    //2
			writeOutTxt();   //3
			MIPWriter.write(ref_mip_resize);  //4
			frameCount_writeOut = frameNum;
		}
	}
	//testSavetxt.close();
	MIPWriter.release();
	outTXT.close();
	timeTXT.close();

	cout << "write out thread say goodbye!!" << endl;
	return;
}

void Experiment::imgProcess()
{
	//Timer time;
	cout << "Image recon and regis thread Say Hello!! :)" << endl;

	int frameCount_imgprocess = 0;
	while (!UserWantToStop)
	{
		if (frameCount_imgprocess != frameNum)
		{
			//time.start();
			ImgReconAndRegis();
			getReconMIP();
			frameCount_imgprocess = frameNum;
			cout << " "; 
		}

	}
	cout << "Image recon and regis thread say goodbye!!" << endl;
	return;
}



void Experiment::controlExp()
{
	Timer laserTimer;
	Timer intervalTimer;
	bool laserTimerStart = false;
	bool intervalTimerStart = false;
	while (!UserWantToStop)
	{
		getParamsFromGUI();
		drawGUIimg();
		if (params.circleOn)   //Periodic stimulation
		{
			if (params.laserOn)
			{
				if (!laserTimerStart)
				{
					laserTimer.start();
					laserTimerStart = true;
				}
				if (laserTimer.getElapsedTimeInMilliSec() > params.laserTime)
				{
					params.laserOn = 0;
					laserTimer.stop();
					laserTimerStart = false;
					cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
					circleCount++;
					cout << endl << "circle Count: " << circleCount << "/" << params.circleNum << endl;
				}
			}
			else
			{
				if (!intervalTimerStart)
				{
					intervalTimer.start();
					intervalTimerStart = true;
				}
				if (intervalTimer.getElapsedTimeInSec() > params.interval)
				{
					params.laserOn = 1;
					intervalTimer.stop();
					intervalTimerStart = false;
					cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
				}
			}
		}
		else    //Manual stimulation
		{
			if (params.laserOn)
			{
				if (!laserTimerStart)
				{
					laserTimer.start();
					laserTimerStart = true;
				}

				if (laserTimer.getElapsedTimeInMilliSec() > params.laserTime)
				{
					params.laserOn = 0;
					laserTimer.stop();
					laserTimerStart = false;
					cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
				}
			}
		}

		if (circleCount >= params.circleNum)
		{
			params.circleOn = 0;
			circleCount = 0;
			cv::createTrackbar("circleOn", "Control Panel for Optogenetic", &params.circleOn, 1);
		}

		if (params.circleOn == 0)
		{
			circleCount = 0;
		}
	}

	cout << "control thread say goodbye" << endl;

	return;
}

void Experiment::generateGalvoVotages()
{
	bool inverse = false;
	cv::Point p(0, 0);
	//continue read pixel coordinates
	galvoVotagesPairs.clear();

	//ROIpoints
	//fishImgProc.FishReg.RegionCoord
	for (int i = 0; i < ROIpoints.size(); i=i+1)
	{
		if (ROIpoints[i].x < 0 || ROIpoints[i].y < 0)
		{
			break;
		}
		cv::Point p = cv::Point(ROIpoints[i].x, ROIpoints[i].y);
		float* Xdata = galvoMatrixX.ptr<float>(p.x);
		float galvo_x = Xdata[p.y];
		float* Ydata = galvoMatrixY.ptr<float>(p.x);
		float galvo_y = Ydata[p.y];
		galvoVotagesPairs.push_back(cv::Point2f(galvo_x, galvo_y));
		
	}



}

void Experiment::galvoControl()
{
	cout << "I'm galvo thread." << endl;


	while (!UserWantToStop)
	{

		bool read_inverse = false;
		while (params.laserOn||params.laserOnLongTime)
		{
			if (!fishMoving)    //Fish are not stimulated during strenuous exercise
			{
				if (read_inverse)
				{
					for (long a = galvoVotagesPairs.size() - 1; a >= 0; a--)
					{
						galvo.spinGalvo(galvoVotagesPairs[a]);
					}
					read_inverse = false;
				}
				else
				{
					for (long a = 0; a < galvoVotagesPairs.size(); a++)
					{
						galvo.spinGalvo(galvoVotagesPairs[a]);
					}
					read_inverse = true;
				}
			}
			else
			{
				galvo.spinGalvo(cv::Point(-2.5, -5));
			}
		}

		galvo.spinGalvo(cv::Point(-2.5, -5));   //out of view
	}

	cout << "galvo thread say goodbye!!" << endl;

	return;
}


void Experiment::TCPconnect()
{
	cout << "I'm TCP server thread!!" << endl;
	server.initialize();
	while (!UserWantToStop)
	{

		if (server.waitingConnect())
		{
			while (!UserWantToStop)
			{
				server.sendData();
				int isOK = server.receive();
				if (server.data > 0 && server.data <= 360)
				{
					fishImgProc.rotationAngleX = server.data;
					isFishMoving(server.data);    //Use the angle passed by tracking
				}
				if (!isOK)
				{
					printf("lose client!!\n");
					//printf("wait for a new connection!!\n");
					cout << endl;
					closesocket(server.socketConn2);
					break;
				}
			}
		}
	}

	server.close();
	cout << "TCP server thread say goodbye!!" << endl;
	return;
}


void Experiment::generateGalvoVoltagesPairs()
{
	while (!UserWantToStop)
	{
		generateGalvoVotages();
		Sleep(1);
	}

	return;
}


void Experiment::isFishMoving(int angle)
{
	headingAngleVec.push_back(angle);
	if (headingAngleVec.size() > 50)  
	{
		vector<int>::iterator k = headingAngleVec.begin();
		headingAngleVec.erase(k);
	}

	double sum = std::accumulate(std::begin(headingAngleVec), std::end(headingAngleVec), 0.0);
	double mean = sum / headingAngleVec.size(); //mean

	double accum = 0.0;
	std::for_each(std::begin(headingAngleVec), std::end(headingAngleVec), [&](const double d)
	{
		accum += (d - mean)*(d - mean);
	});

	double stdev = sqrt(accum / (headingAngleVec.size() - 1)); //Variance

	if (abs(angle - mean) > 5 || stdev > 5) 
	{
		cout << "fish move" << endl;
		fishMoving = true;
	}
	else
	{
		fishMoving = false;
	}

	return;
}



void Experiment::clear()
{
	outTXT.close();
	fishImgProc.freeMemory();

	delete[] Image;
	delete[] Image4bin;
	delete[] mip_cpu;
	delete[] cropResult_cpu;

	cv::destroyAllWindows();

	T2Cam_TurnOff(cam_data, cam_handle);
	T2Cam_CloseLib();

	cv::Point pt(0, 0);
	galvo.spinGalvo(pt);

	cout << "clear exp" << endl;
	return;
}



void Experiment::exit()
{
	exit();
	return;
}
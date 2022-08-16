#include"experiment.h"
#include"kexinLibs.h"
#include "gdal_priv.h"
#include <gdal.h>

#include"Timer.h"

#include  <io.h>  
#include  <stdio.h>  
#include  <stdlib.h> 

#include"avir.h"

#include <opencv2/core/utils/logger.hpp>

using namespace std;

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
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);  //关闭opencv的警告

	Image = new unsigned short int[2048 * 2048 * 1];  //开辟缓存区
	Image_forSave = new unsigned short int[2048 * 2048 * 1];
	Image4bin = new unsigned short int[512 * 512 * 1];
	mip_cpu = new float[200 * 200 * 1];
	cropResult_cpu = new float[76 * 95 * 50]();

	MIP = cv::Mat(200, 200, CV_32FC1, mip_cpu);

	////初始化相机
	cam_data = T2Cam_CreateCamData(); //动态申请CamData结构体的空间,创建指向该空间的cam_data指针
	T2Cam_InitializeLib(&cam_handle);
	SetupBinningandAOI(cam_handle);
	T2Cam_InitializeCamData(cam_data, cam_handle);
	getUserSettings(cam_handle);
	CreateBuffer(cam_data, cam_handle);
	cout << "camera prepare done" << endl;

	////初始化galvo
	galvoXmin = -10;
	galvoYmin = -10;
	galvo.initialize();
	generateMatFromYaml(galvoMatrixX, "GalvoX_matrix");
	generateMatFromYaml(galvoMatrixY, "GalvoY_matrix");
	//resize(galvoMatrixX, galvoMatrixX, cv::Size(), scale, scale);
	//resize(galvoMatrixY, galvoMatrixY, cv::Size(), scale, scale);
	//	galvoMatrixX = Mat(galvoMatrixX, Rect(180, 180, 360, 360));
	//	galvoMatrixY = Mat(galvoMatrixY, Rect(180, 180, 360, 360));
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
	UserWantToStop = 0;

	return;
}

void Experiment::readFullSizeImgFromFile()
{
	cout << "read Image thread say hello!!" << endl;

	string beforeResizeImgPath = "D:/kexin/Online-Zebrafish-Optogenetic/data/r20210824_X10";
	vector<string> beforeResizeImgNames;
	getFileNames(beforeResizeImgPath, beforeResizeImgNames);

	while (!UserWantToStop)
	{
		string filename = beforeResizeImgNames[frameNum];
		GDALDataset *poDataset;   //GDAL数据集
		poDataset = (GDALDataset *)GDALOpen(filename.data(), GA_ReadOnly);
		if (poDataset == NULL)
		{
			cout << "fail in open files!!!" << endl;
			//return;
			continue;
		}
		int nImgSizeX = poDataset->GetRasterXSize();
		int nImgSizeY = poDataset->GetRasterYSize();
		GDALRasterBand *poBand;
		int bandcount = poDataset->GetRasterCount();	// 获取波段数
		//unsigned short int *Image = new unsigned short int[nImgSizeX*nImgSizeY*bandcount];  //开辟缓存区
		if (DEBUG)
		{
			cout << nImgSizeX << "  *  " << nImgSizeY << endl;
			cout << "波段数：" << bandcount << endl;
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
					//cout << Image[(bandind - 1) * nImgSizeX * nImgSizeY + i * nImgSizeY + j] << endl;
				}
			}
			delete[] pafScanline;
		}

		//delete poDataset;

		resizeImg();

		cout << "read img done: " << filename << endl;
		frameNum = frameNum + 1;

		if (frameNum >= beforeResizeImgNames.size()-1)
		{
			frameNum = 0;
			//UserWantToStop = 1;
			//break;
		}
	}

	cout << "read Image thread say goodbye!!" << endl;


	return;
}


void Experiment::readFullSizeImgFromCamera()
{
	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle);
	while (!UserWantToStop)
	{
		if (T2Cam_GrabFrame(cam_data, cam_handle) == 0)
		{
			memcpy(Image, cam_data->ImageRawData, CCDSIZEX *CCDSIZEY * sizeof(unsigned short));
		}
		else
		{
			cout << "Error in imaging acquisition!" << endl;
			break;
		}

		preProcessImg();
		resizeImg();
		frameNum = frameNum + 1;
	}

	AT_Command(cam_handle, L"AcquisitionStop");
	cout << endl << endl;

	return;
}

void Experiment::preProcessImg()
{
	cv::Mat temp(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1, Image);
	cv::flip(temp, temp, 0);   //flipcode=0,垂直翻转图像
	cv::threshold(temp, temp, thre, maxValue, CV_THRESH_TRUNC);   //高于thre的数据被置为thre

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
	fishImgProc.reconImage();//重构读进来的图像
	fishImgProc.cropReconImage();
	//rotation
	fishImgProc.matchingANDrotationXY();
	//crop
	fishImgProc.cropRotatedImage(params.xbias, params.ybias);
	////crop的结果构建movingTensor，和fixTensor一起送入网络处理
	fishImgProc.libtorchModelProcess();
	////结合rotation/crop/affine的数据做坐标转换
	ROIpoints = fishImgProc.ZBB2FishTransform(roi);

	return;
}

//
////改成可存储任意大小的图像
//void Experiment::saveImg2Disk(string filename)
//{
//	memcpy(Image_forSave, Image, CCDSIZEX *CCDSIZEY * sizeof(unsigned short));
//	//输出图像
//	GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
//	GDALDataset *ds = pDriver->Create(filename.c_str(), 2048, 2048, 1, GDT_UInt16, NULL);
//	if (ds == NULL)
//	{
//		cout << "Failed to create output file!" << endl;
//		system("pause");
//		return;
//	}
//	unsigned short int *writeOut_buffer = new unsigned short int[2048];
//	for (int band = 0; band < 1; band++)
//	{
//		for (int i = 0; i < 2048; i++)//行循环
//		{
//			for (int j = 0; j < 2048; j++)//列循环
//			{
//				writeOut_buffer[j] = Image_forSave[band * 2048 * 2048 + i * 2048 + j];
//			}
//			ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, i, 2048, 1, writeOut_buffer, 2048, 1, GDT_UInt16, 0, 0);
//		}
//		//cout << band << endl;
//	}
//	delete[] writeOut_buffer;
//	//delete ds;
//	//delete pDriver;
//
//	return;
//}

void Experiment::getReconMIP()
{
	cudaMemcpy(mip_cpu, fishImgProc.image2D_XY_gpu, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);
	//MIP = cv::Mat(200, 200, CV_32FC1, mip_cpu);//重复创建mat导致内存泄露,画出来的MIP是正确的
	//float val = MIP.at<float>(100, 100);
	//cout << val << endl;

	//MIP.data = (uchar*)mip_cpu;   //用这个方法赋值，内存不再泄露,但是画出来的MIP全是黑的
	//memcpy(MIP.ptr<float>(0), mip_cpu, sizeof(float) * 200 * 200 * 1); //用这个方法赋值，内存不再泄露,但是画出来的MIP全是黑的

	cv::normalize(MIP, MIP, 0, 1, cv::NormTypes::NORM_MINMAX);

	return;
}

void Experiment::setupGUI()
{
	//set up GUI
	cout << "creating control panel for optogenetic..." << endl;
	cv::namedWindow("Control Panel for Optogenetic", cv::WINDOW_NORMAL);
	cv::resizeWindow("Control Panel for Optogenetic", 700, 500);

	cv::createTrackbar("recordOn", "Control Panel for Optogenetic", &recordOn, 1);
	cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
	cv::createTrackbar("ScanTime", "Control Panel for Optogenetic", &params.laserTime, 5000);
	cv::createTrackbar("xsize", "Control Panel for Optogenetic", &params.xsize, 20);
	cv::createTrackbar("ysize", "Control Panel for Optogenetic", &params.ysize, 20);
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
	//refresh GalvoOn
	//cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
	//get params from GUI
	recordOn = cv::getTrackbarPos("recordOn", "Control Panel for Optogenetic");
	params.laserOn = cv::getTrackbarPos("LaserOn", "Control Panel for Optogenetic");
	params.laserTime = cv::getTrackbarPos("ScanTime", "Control Panel for Optogenetic");
	params.xsize = cv::getTrackbarPos("xsize", "Control Panel for Optogenetic");
	params.ysize = cv::getTrackbarPos("ysize", "Control Panel for Optogenetic");
	params.xpos = cv::getTrackbarPos("xpos", "Control Panel for Optogenetic");
	params.ypos = cv::getTrackbarPos("ypos", "Control Panel for Optogenetic");
	params.xbias = cv::getTrackbarPos("xbias", "Control Panel for Optogenetic");
	params.ybias = cv::getTrackbarPos("ybias", "Control Panel for Optogenetic");

	UserWantToStop = cv::getTrackbarPos("Stop", "Control Panel for Optogenetic");


	//防止越界
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

	//准备reference图像，ZBB
	cv::Mat ref = cv::imread("Ref-zbb-MIP.png", 0);   //0：read gray image
	cv::rectangle(ref, roi, cv::Scalar(255), 2);
	cv::resize(ref, ref_resize, cv::Size(ref.cols * 2, ref.rows * 2));
	cv::Mat refROI = ref_MIP(cv::Rect(0, 0, ref_resize.cols, ref_resize.rows));
	ref_resize.copyTo(refROI);


	//cv::imshow("ttt", MIP);

	//准备当前帧的MIP
	cv::Mat MIP_8u = MIP.clone();
	MIP_8u = (MIP_8u * 255);
	MIP_8u.convertTo(MIP_8u, CV_8UC1);
	//cout << "ROIpoints: " << ROIpoints.size() << endl;
	if (ROIpoints.size() > 3)
	{
		cv::RotatedRect re = cv::minAreaRect(ROIpoints);
		//获取旋转矩形的四个顶点
		cv::Point2f* vertices = new cv::Point2f[4];
		re.points(vertices);

		//逐条边绘制
		for (int j = 0; j < 4; j++)
		{
			cv::line(MIP_8u, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255),2);
		}
	}
	//for (int i = 0; i < ROIpoints.size(); i++)
	//{
	//	cv::Point3f p = ROIpoints[i];
	//	cv::circle(MIP_8u, cv::Point(p.x, p.y), 1, cv::Scalar(255));
	//}


	cv::Mat MIPROI = ref_MIP(cv::Rect(200, 0, MIP_8u.cols, MIP_8u.rows));
	MIP_8u.copyTo(MIPROI);


	cv::Mat ref_mip_resize;
	cv::resize(ref_MIP, ref_mip_resize, cv::Size(ref_MIP.cols * 2, ref_MIP.rows * 2));

	cv::imshow("test", ref_mip_resize);


	cv::waitKey(1);

	return;
}


void Experiment::initializeWriteOut()
{
	//初始化GDLA
	GDALAllRegister(); OGRRegisterAll();
	//设置支持中文路径
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

	txtName = path + fishName + "/"+ fishName +".txt";
	outTXT.open(txtName);

	return;
}

void Experiment::writeOutTxt()
{
	outTXT << "frameNum:" << frameNum << endl;
	outTXT << "laserOn:" << params.laserOn << ", xsize:" << params.xsize << ", xpos:" << params.xpos
		<< ", ysize:" << params.ysize << ", ypos:" << params.ypos << endl;
	outTXT << "rotationAngleX:" << fishImgProc.rotationAngleX << endl;
	outTXT << "rotationAngleY:" << fishImgProc.rotationAngleY << endl;
	outTXT << "cropPoint:" << fishImgProc.cropPoint << endl;
	outTXT << "Moving2FixAffineMatrix:";
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
		if (recordOn)
		{
			if (frameCount_writeOut != frameNum)
			{
				//saveImg2Disk(folderName + "/" + int2string(6, frameNum) + ".tif");
				saveAndCheckImage(Image, CCDSIZEX, CCDSIZEY, 1, rawfolderName + "/" + int2string(6, frameNum) + ".tif");
				cudaMemcpy(cropResult_cpu, fishImgProc.ObjCropRed_gpu, sizeof(float) * 76 * 95 * 50, cudaMemcpyDeviceToHost);
				saveAndCheckImage(cropResult_cpu, 76, 95, 50, cropfolderName + "/" + int2string(6, frameNum) + ".tif");
				writeOutTxt();
				frameCount_writeOut = frameNum;
			}
		}
	}

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
			cout << " ";   //这里必须有一句输出？？？
			//time.stop();
			//cout << "process time: " << time.getElapsedTimeInMilliSec() << endl;
		}

	}
	cout << "Image recon and regis thread say goodbye!!" << endl;
	return;
}



void Experiment::controlExp()
{
	Timer time;
	bool timerStart = false;
	while (!UserWantToStop)
	{
		getParamsFromGUI();
		drawGUIimg();

		if (params.laserOn)
		{
			if (!timerStart)
			{
				time.start();
				timerStart = true;
			}

			if (time.getElapsedTimeInMilliSec() > params.laserTime)
			{
				params.laserOn = 0;
				time.stop();
				timerStart = false;
				cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);

			}
		}

		//cv::waitKey(1);
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
	for (int i = 0; i < ROIpoints.size(); i++)
	{
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
		while (params.laserOn)
		{
			generateGalvoVotages();
			if (read_inverse)
			{
				for (long a = galvoVotagesPairs.size() - 1; a >= 0; a--)
				{
					galvo.spinGalvo(galvoVotagesPairs[a]);
					//cout << galvoVotagesPairs[a] << endl;
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
		galvo.spinGalvo(cv::Point(-5, -5));   //放在一个很偏的点
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
				int isOK = server.receive();
				if (server.data >0&&server.data<=360)
				{
					fishImgProc.rotationAngleX = server.data;
					//cout << fishImgProc.rotationAngleX << endl;
				}
				//fishImgProc.rotationAngleX = server.data;
				//cout << server.data << endl;
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



void Experiment::clear()
{
	outTXT.close();
	fishImgProc.freeMemory();

	delete[] Image;
	delete[] Image_forSave;
	delete[] Image4bin;
	delete[] mip_cpu;
	delete[] cropResult_cpu;

	cv::destroyAllWindows();

	//相机, SDK, 释放内存
	//Camera and Libs
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



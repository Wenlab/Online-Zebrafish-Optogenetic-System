#include"experiment.h"

#include "gdal_priv.h"
#include <gdal.h>

#include"avir.h"

using namespace std;

Experiment::Experiment(std::string modle_path) :fishImgProc(modle_path)
{

}

Experiment::~Experiment()
{

}

void Experiment::prepareMemory()
{
	Image = new unsigned short int[2048*2048*1];  //开辟缓存区
	Image4bin = new unsigned short int[512 * 512 * 1];


	return;
}

void Experiment::initialize()
{
	fishImgProc.initialize();
	cout << "initialize fishImgProc done" << endl;

	MIP = cv::Mat(200, 200, CV_32FC1);


	return;
}

void Experiment::readFullSizeImgFromFile(string filename)
{
	GDALDataset *poDataset;   //GDAL数据集
	poDataset = (GDALDataset *)GDALOpen(filename.data(), GA_ReadOnly);
	if (poDataset == NULL)
	{
		cout << "fail in open files!!!" << endl;
		return;
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

	}

	cout << "read img done: " << filename << endl;
}

void Experiment::readFullSizeImgFromCamera()
{

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
	fishImgProc.cropRotatedImage();
	////crop的结果构建movingTensor，和fixTensor一起送入网络处理
	fishImgProc.libtorchModelProcess();
	////结合rotation/crop/affine的数据做坐标转换
	ROIpoints = fishImgProc.ZBB2FishTransform(roi);

	return;
}

void Experiment::saveImg2Disk(string filename)
{
	//输出图像
	GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset *ds = pDriver->Create(filename.c_str(), 2048, 2048, 1, GDT_UInt16, NULL);
	if (ds == NULL)
	{
		cout << "Failed to create output file!" << endl;
		system("pause");
		return;
	}
	unsigned short int *writeOut_buffer = new unsigned short int[2048];
	for (int band = 0; band < 1; band++)
	{
		for (int i = 0; i < 2048; i++)//行循环
		{
			for (int j = 0; j < 2048; j++)//列循环
			{
				writeOut_buffer[j] = Image[band * 2048 * 2048 + i * 2048 + j];
			}
			ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, i, 2048, 1, writeOut_buffer, 2048, 1, GDT_UInt16, 0, 0);
		}
		//cout << band << endl;
	}
	free(writeOut_buffer);
	delete ds;

	return;
}

void Experiment::getReconMIP()
{
	float* temp = new float[200 * 200 * 1];
	cudaMemcpy(temp, fishImgProc.image2D_XY_gpu, sizeof(float) * 200 * 200 * 1, cudaMemcpyDeviceToHost);
	MIP = cv::Mat(200, 200, CV_32FC1, temp);

	cv::normalize(MIP, MIP, 0, 1, cv::NormTypes::NORM_MINMAX);

	return;
}

void Experiment::setupGUI()
{
	//set up GUI
	cout << "creating control panel for optogenetic..." << endl;
	cv::namedWindow("Control Panel for Optogenetic", cv::WINDOW_NORMAL);
	cv::resizeWindow("Control Panel for Optogenetic", 600, 400);

	cv::createTrackbar("recordOn", "Control Panel for Optogenetic", &recordOn, 1);
	cv::createTrackbar("LaserOn", "Control Panel for Optogenetic", &params.laserOn, 1);
	cv::createTrackbar("ScanTime", "Control Panel for Optogenetic", &params.laserTime, 5000);
	cv::createTrackbar("xsize", "Control Panel for Optogenetic", &params.xsize, 20);
	cv::createTrackbar("ysize", "Control Panel for Optogenetic", &params.ysize, 20);
	cv::createTrackbar("xpos", "Control Panel for Optogenetic", &params.xpos, 50);
	cv::createTrackbar("ypos", "Control Panel for Optogenetic", &params.ypos, 50);
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

	UserWantToStop = cv::getTrackbarPos("Stop", "Control Panel for Optogenetic");

	roi = cv::Rect(params.xpos, params.ypos, params.xsize, params.ysize);

	cv::waitKey(1);
	return;
}


void Experiment::drawGUIimg()
{

	ref = cv::imread("Ref-zbb-MIP.png", 0);   //0：read gray image

	cv::rectangle(ref, roi, cv::Scalar(255), 2);

	ref_MIP = cv::Mat(200, 400, CV_8UC1, cv::Scalar(0));
	cv::resize(ref, ref_resize, cv::Size(ref.cols * 2, ref.rows * 2));

	cv::Mat refROI = ref_MIP(cv::Rect(0, 0, ref_resize.cols, ref_resize.rows));
	ref_resize.copyTo(refROI);

	cv::Mat MIP_8u = MIP.clone();
	MIP_8u = (MIP_8u * 255);
	MIP_8u.convertTo(MIP_8u, CV_8UC1);

	for (int i = 0; i < ROIpoints.size(); i++)
	{
		cv::Point3f p = ROIpoints[i];
		cv::circle(MIP_8u, cv::Point(p.x, p.y), 1, cv::Scalar(255));
	}


	cv::Mat MIPROI = ref_MIP(cv::Rect(200, 0, MIP_8u.cols, MIP_8u.rows));
	MIP_8u.copyTo(MIPROI);

	cv::Mat ref_mip_resize;
	cv::resize(ref_MIP, ref_mip_resize, cv::Size(ref_MIP.cols * 2, ref_MIP.rows * 2));

	cv::imshow("test", ref_mip_resize);

	cv::waitKey(1);

	return;
}


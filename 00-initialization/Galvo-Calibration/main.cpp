#include"Talk2Camera.h"
#include"Talk2Galvo.h"
#include"Timer.h"

//GDAL
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>

#include <io.h>
#include <sstream>
#include <direct.h> //_mkdir fun

using namespace std;
using namespace cv;


void saveAndCheckImage(unsigned short int* imageData, int col_total, int row_total, int z_total, string name);
void preProcessImg(unsigned short int *Image);
std::string getTime();
std::string int2string(int n, int i);
std::string Float2Str(float Num);


int main()
{
	//Input voltage range of galvo -1.5~1.5

	float galvoMin = -1.5;
	float galvoMax = 1.5;
	float step = 0.2;

	//cameras
	AT_H cam_handle; 
	CamData* cam_data;
	//Initializing the camera
	cam_data = T2Cam_CreateCamData(); 
	T2Cam_InitializeLib(&cam_handle);
	SetupBinningandAOI(cam_handle);
	T2Cam_InitializeCamData(cam_data, cam_handle);
	getUserSettings(cam_handle);
	CreateBuffer(cam_data, cam_handle);
	cout << "camera prepare done" << endl;

	//Initializing the galvo
	GalvoData galvo;
	galvo.initialize();

	//Initializing GDLA
	GDALAllRegister(); OGRRegisterAll();
	//Set Chinese path support
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");

	unsigned short int *Image = new unsigned short int[2048 * 2048 * 1];

	//Initializing save path
	int ret;
	string rootPath = "D:/kexin/Galvo-Calibration/calibrationData/";
	ret = _access(rootPath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(rootPath.c_str());
	}

	string time = getTime();
	rootPath = rootPath + time + "/";
	ret = _access(rootPath.c_str(), 0);
	if (ret != 0)
	{
		_mkdir(rootPath.c_str());
	}

	//Start Acquisition
	T2Cam_StartAcquisition(cam_handle);
	for (float i = galvoMin; i < galvoMax; i = i + step)
	{
		for (float j = galvoMin; j < galvoMax; j = j + step)
		{
			string imgFolderName = rootPath + "GalvoX_" + Float2Str(i) + "_GalvoY_" + Float2Str(j) + "/";
			ret = _access(imgFolderName.c_str(), 0);
			if (ret != 0)
			{
				_mkdir(imgFolderName.c_str());
			}


			galvo.spinGalvo(cv::Point2f(i, j));
			cout << "GalvoX: " << i << "   GalvoY: " << j << "   ";

			Sleep(100);
			for (int aa = 0; aa < 10; aa++)
			{
				if (T2Cam_GrabFrame(cam_data, cam_handle) == 0)
				{
					memcpy(Image, cam_data->ImageRawData, CCDSIZEX * CCDSIZEY * sizeof(unsigned short));
					cv::Mat temp(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1, Image);
					cv::flip(temp, temp, 0);   //flipcode=0, Vertical Flip Image
				}
				else
				{
					cout << "Error in imaging acquisition!" << endl;
					break;
				}
				//write out
				string imgName = imgFolderName + int2string(3, aa) + ".tif";
				saveAndCheckImage(Image, CCDSIZEX, CCDSIZEY, 1, imgName);
				cout << aa << " ";
			}
			cout << endl;
		}
	}

	delete[] Image;
	return 0;
}


void preProcessImg(unsigned short int *Image)
{
	cv::Mat temp(cv::Size(CCDSIZEX, CCDSIZEY), CV_16UC1, Image);
	cv::flip(temp, temp, 0);   //flipcode=0, Vertical Flip Image

	return;
}


void saveAndCheckImage(unsigned short int* imageData, int col_total, int row_total, int z_total, string name)
{

	GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset *ds = pDriver->Create(name.c_str(), col_total, row_total, z_total, GDT_UInt16, NULL);
	if (ds == NULL)
	{
		cout << "Failed to create output file!" << endl;
		system("pause");
		return;
	}

	unsigned short int  *ObjRecon_buffer = new unsigned short int[col_total];
	for (int band = 0; band < z_total; band++)
	{
		for (int i = 0; i < row_total; i++)//row
		{
			for (int j = 0; j < col_total; j++)//col
			{
				ObjRecon_buffer[j] = imageData[band * col_total * row_total + i * col_total + j];
			}
			ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, i, col_total, 1, ObjRecon_buffer, col_total, 1, GDT_UInt16, 0, 0);
		}
	}
	delete ds;
	delete[] ObjRecon_buffer;


	return;
}


std::string getTime()
{
	time_t nowtime;
	nowtime = time(NULL);

	tm local;
	localtime_s(&local, &nowtime);

	char buf[80];
	strftime(buf, 80, "%Y%m%d_%H%M", &local);
	cout << buf << endl;
	std::string time = buf;
	return time;
}


std::string Float2Str(float Num)
{
	std::ostringstream oss;
	oss << Num;
	std::string str(oss.str());
	return str;
}


std::string int2string(int n, int i)
{
	char s[BUFSIZ];
	sprintf_s(s, "%d", i);
	int l = strlen(s);

	if (l > n)
	{
		std::cout << "The length of the integer is greater than the length of the string to be formatted!";
	}
	else
	{
		std::stringstream M_num;
		for (int i = 0; i < n - l; i++)
			M_num << "0";
		M_num << i;

		return M_num.str();
	}
}

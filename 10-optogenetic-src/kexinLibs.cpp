#include"kexinLibs.h"

//GDAL
#include "gdal_alg.h";
#include "gdal_priv.h"
#include <gdal.h>

//Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>
#define _USE_MATH_DEFINES
#include <math.h>

#include <io.h>
#include <sstream>
#include <direct.h> //_mkdir fun

using namespace std;



std::vector<float> rescaleAffineMatrix(std::vector<float> v)
{


	v[0] = v[0] / 100 / 10 + 1;
	v[1] = v[1] / 100;
	v[2] = v[2] / 100;
	v[3] = v[3] / 100;
	v[4] = v[4] / 100 / 10 + 1;
	v[5] = v[5] / 100;
	v[6] = v[6] / 100;
	v[7] = v[7] / 100;
	v[8] = v[8] / 100 / 10 + 1;
	v[9] = v[9] / 100 * 77;
	v[10] = v[10] / 100 * 95;
	v[11] = v[11] / 100 * 52;

	return v;
}


float* readImgFromFile(string filename)
{
	GDALAllRegister(); OGRRegisterAll();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	GDALDataset *poDataset;  
	poDataset = (GDALDataset *)GDALOpen(filename.data(), GA_ReadOnly);
	if (poDataset == NULL)
	{
		cout << "fail in open files!!!" << endl;
		return 0;
	}
	int nImgSizeX = poDataset->GetRasterXSize();
	int nImgSizeY = poDataset->GetRasterYSize();
	GDALRasterBand *poBand;
	int bandcount = poDataset->GetRasterCount();	
	float *Image = new float[nImgSizeX*nImgSizeY*bandcount];  
	unsigned short int maxfix = 0;
	unsigned short int minfix = 10000;

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
				Image[(bandind - 1) * nImgSizeX * nImgSizeY + i * nImgSizeY + j] = float(pafScanline[i*nImgSizeY + j]);
			}
		}

	}

	cout << "read img done: " << filename << endl;
	return Image;
}


torch::Tensor normalizeTensor(torch::Tensor tensor)
{
	auto max_result = torch::max(tensor);
	auto min_result = torch::min(tensor);


	//normalize
	tensor = (tensor - min_result) / (max_result - min_result);

	tensor = tensor.unsqueeze(0);

	return tensor;
}


std::vector<float> rescaleAffineMatrix(torch::Tensor affineMatrix)
{
	//affineMatrix: 1*12
	affineMatrix = affineMatrix.squeeze(0);
	affineMatrix = affineMatrix / 100.0;


	affineMatrix = affineMatrix.to(torch::kCPU);
	std::vector<float> v(affineMatrix.data_ptr<float>(), affineMatrix.data_ptr<float>() + affineMatrix.numel());

	v[0] = v[0] / 10 + 1;
	v[4] = v[4] / 10 + 1;
	v[8] = v[8] / 10 + 1;
	v[9] = v[9] * 77;
	v[10] = v[10] * 95;
	v[11] = v[11] * 52;

	return v;
}



//GDALDataset::RasterIO(GDALRWFlag  eRWFlag,
//	int  nXOff,
//	int  nYOff,
//	int  nXSize,
//	int  nYSize,
//	void *  pData,
//	int  nBufXSize,
//	int  nBufYSize,
//	GDALDataType  eBufType,
//	int  nBandCount,
//	int *  panBandMap,
//	int  nPixelSpace,
//	int  nLineSpace,
//	int  nBandSpace
//)
void saveAndCheckImage(float* imageData, int col_total, int row_total, int z_total, string name)
{
	GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset *ds = pDriver->Create(name.c_str(), col_total, row_total, z_total, GDT_Float32, NULL);
	if (ds == NULL)
	{
		cout << "Failed to create output file!" << endl;
		system("pause");
		return;
	}
	for (int band = 0; band < z_total; band++)
	{
		ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, 0, col_total, row_total, imageData+band*row_total*col_total, col_total, row_total, GDT_Float32, 0, 0);

	}
	GDALClose(ds);


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
	for (int band = 0; band < z_total; band++)
	{
		ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, 0, col_total, row_total, imageData+band* row_total*col_total, col_total, row_total, GDT_UInt16, 0, 0);
	}
	GDALClose(ds);
	return;
}

void getFileNames(std::string path, std::vector<std::string>& files)
{
	intptr_t hFile = 0;

	struct _finddata_t fileinfo;
	std::string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{

			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFileNames(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}


std::string int2string(int n, int i)
{
	char s[BUFSIZ];
	sprintf_s(s, "%d", i);
	int l = strlen(s);

	if (l > n)
	{
		std::cout << "整型的长度大于需要格式化的字符串长度！";
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

void generateMatFromYaml(cv::Mat& matrix, std::string dataName)
{
	string fileName = dataName + ".yaml";
	cv::FileStorage file(fileName, cv::FileStorage::READ);
	file[dataName] >> matrix;

	cout << "reading " << dataName << " from " << fileName;
	cout << "Mat:" << endl;
	cout << dataName << ".size(): " << matrix.size() << endl;
	cout << dataName << ".depth: " << matrix.depth() << endl;
	cout << dataName << ".channels: " << matrix.channels() << endl;
	file.release();
}
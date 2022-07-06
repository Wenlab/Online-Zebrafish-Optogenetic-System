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
	std::vector<cv::Point3f> points = fishImgProc.ZBB2FishTransform();

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
}

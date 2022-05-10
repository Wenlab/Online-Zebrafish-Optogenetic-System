#include "zbb2FishImg.h"

using namespace std;
using namespace cv;

#pragma warning(disable : 4996)

zbb2FishImg::zbb2FishImg()
{
}


zbb2FishImg::~zbb2FishImg()
{
}

void zbb2FishImg::initialize()
{
	string zbbPath = "anatomyList_4bin.txt";
	vector<pair<string, Point>> zbbMapPair;
	zbbMapPair = readZBBMapFromTxt(zbbPath);
	zbbMapVec = makeZbbMapVec(zbbMapPair);

	//getRegionFromUser();
}

void zbb2FishImg::getRegionFromUser(cv::Rect Reg)
{
	Region = Reg;

	//将区域转化成坐标
	for (int i = Region.x; i < Region.x + Region.width; i++)
	{
		for (int j = Region.y; j < Region.y + Region.height; j++)
		{
			Point3f temp{ (float)i,(float)j,0 };
			RegionCoord.push_back(temp);
			//cout << temp << endl;
		}
	}

	//查询该区域包含的脑区
	BrainRegionName = queryRegionName(Region);

	return;
}

//获取从ZBB到Fix Image的仿射矩阵
void zbb2FishImg::getZBB2FixAffineMatrix(std::vector<float> Fix2zbbAM)
{
	zbb2FixAM = inverseAffineMatrix(Fix2zbbAM);
	return;
}

//获取从fix Image到moving Image的仿射矩阵
void zbb2FishImg::getFix2MovingAffineMatrix(std::vector<float> Moving2FixAM)
{
	Fix2MovingAM = inverseAffineMatrix(Moving2FixAM);
	return;
}

//获取从旋转后图像到moving Image的crop信息
void zbb2FishImg::getCropPoint(cv::Point3f pt)
{
	cropPoint = pt;
}

//获取从原始图像到旋转后图像的旋转角度信息
//绕Z轴的旋转、绕X轴的旋转
void zbb2FishImg::getRotationMatrix(float rotationAngleZ, float rotationAngleX)
{
	//绕Z轴的旋转矩阵
	rotationAngleZ = rotationAngleZ * PI / 180.0;
	Eigen::Matrix3f rotationMatrixZ;
	rotationMatrixZ << cos(rotationAngleZ), -sin(rotationAngleZ), 0,
		sin(rotationAngleZ), cos(rotationAngleZ), 0,
		0, 0, 1;
	//绕X轴的旋转矩阵
	rotationAngleX = rotationAngleX * PI / 180.0;
	Eigen::Matrix3f rotationMatrixX;
	rotationMatrixX << 1, 0, 0,
		0, cos(rotationAngleX), -sin(rotationAngleX),
		0, sin(rotationAngleX), cos(rotationAngleX);

	//两个旋转矩阵相乘求总旋转矩阵
	rotationMatrix = rotationMatrixZ;
	//cout << rotationMatrix << endl << endl;
	return;
}


//将区域从ZBB上的选定坐标还原到fish
//使用上面几个函数得到的信息
void zbb2FishImg::ZBB2FishTransform()
{
	for (int i = 0; i < RegionCoord.size(); i++)
	{
		Point3f p;
		Point3f temp = RegionCoord[i];
		p = applyAffineMatrixOn3DCoord(temp, zbb2FixAM);
		p = applyAffineMatrixOn3DCoord(p, Fix2MovingAM);
		p = p + cropPoint;
		p = applyRotationMatrixOn3DCoord(p, rotationMatrix);

		regionInFish.push_back(p);
	}

	return;
}


vector<pair<string, Point>> zbb2FishImg::readZBBMapFromTxt(string file)
{
	vector<pair<string, Point>> zbbMap;

	ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 

	string str, pattern;
	pattern = ",";
	while (getline(infile, str))
	{
		char * strc = new char[strlen(str.c_str()) + 1];
		strcpy(strc, str.c_str());
		vector<string> res;
		char* temp = strtok(strc, pattern.c_str());
		while (temp != NULL)
		{
			res.push_back(string(temp));
			temp = strtok(NULL, pattern.c_str());
		}
		delete[] strc;

		pair<string, Point> t{ res[0],Point(atof(res[1].c_str()),atof(res[2].c_str())) };
		zbbMap.push_back(t);
		res.clear();
		//cout << str << endl;
	}
	infile.close();             //关闭文件输入流 

	return zbbMap;
}

vector<vector<vector<string>>> zbb2FishImg::makeZbbMapVec(vector<pair<string, Point>> zbbMapPair)
{
	int a = 77;
	int b = 95;
	vector<vector<vector<string>>> zbbMapVec(a, vector<vector<string>>(b));

	//cout << zbbMapPair.size() << endl;
	for (int i = 0; i < zbbMapPair.size(); i++)
	{
		pair<string, Point> temp = zbbMapPair[i];
		zbbMapVec[temp.second.x - 1][temp.second.y - 1].push_back(temp.first);   //matlab保存的坐标是从1开始
	}
	//去除每个像素位置的相同元素
	for (int i = 0; i < zbbMapVec.size(); i++)
	{
		for (int j = 0; j < zbbMapVec[i].size(); j++)
		{
			zbbMapVec[i][j].erase(unique(zbbMapVec[i][j].begin(), zbbMapVec[i][j].end()), zbbMapVec[i][j].end());
			//cout << zbbMapVec[i][j].size() << endl;
		}
	}
	return zbbMapVec;
}

vector<string> zbb2FishImg::queryRegionName(Rect region)
{
	vector<string> queryRegion;
	for (int i = region.x; i < region.x + region.width; i++)
	{
		for (int j = region.y; j < region.y + region.height; j++)
		{
			vector<string> pixelLable = zbbMapVec[i][j];
			//cout << pixelLable.size() << endl;
			for (int m = 0; m < pixelLable.size(); m++)
			{
				queryRegion.push_back(pixelLable[m]);
			}
		}
	}

	//删除重复的区域名称
	sort(queryRegion.begin(), queryRegion.end());
	queryRegion.erase(unique(queryRegion.begin(), queryRegion.end()), queryRegion.end());

	return queryRegion;
}


void zbb2FishImg::clear()
{
	BrainRegionName.clear();
	RegionCoord.clear();

	regionInFish.clear();
	return;
}



std::vector<float> inverseAffineMatrix(std::vector<float> am)
{

	Eigen::Matrix3f am_eigen;  //不包含三个平移参数
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_eigen(b, a) = am[a * 3 + b];
		}
	}
	Eigen::Matrix3f am_eigen_inverse = am_eigen.inverse();   //求出的逆和原矩阵相乘结果不为1？
	//std::cout << am_eigen_reverse * am_eigen << std::endl;

	Eigen::Vector3f am_trans;
	am_trans << am[9], am[10], am[11];
	Eigen::Vector3f am_trans_inverse = am_trans.transpose() * (-am_eigen_inverse);


	//std::cout << am_trans << std::endl <<std::endl<< am_trans_inverse << std::endl;
	std::vector<float> am_inverse(12);
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_inverse[a * 3 + b] = am_eigen_inverse(b, a);
		}
	}

	am_inverse[9] = am_trans_inverse(0);
	am_inverse[10] = am_trans_inverse(1);
	am_inverse[11] = am_trans_inverse(2);

	return am_inverse;
}

Point3f applyAffineMatrixOn3DCoord(Point3f temp, vector<float> am)
{
	Point3f p;
	p.x = temp.x*am[0] + temp.y*am[1] + temp.z*am[2] + am[9];
	p.y = temp.x*am[3] + temp.y*am[4] + temp.z*am[5] + am[10];
	p.z = temp.x*am[6] + temp.y*am[7] + temp.z*am[8] + am[11];

	return p;
}

Point3f applyRotationMatrixOn3DCoord(Point3f temp, Eigen::Matrix3f rotationMatrix)
{
	Point3f bias(100, 100, 25);
	temp = temp - bias;
	Point3f p;
	p.x = temp.x*rotationMatrix(0, 0) + temp.y*rotationMatrix(0, 1) + temp.z*rotationMatrix(0, 2);
	p.y = temp.x*rotationMatrix(1, 0) + temp.y*rotationMatrix(1, 1) + temp.z*rotationMatrix(1, 2);
	p.z = temp.x*rotationMatrix(2, 0) + temp.y*rotationMatrix(2, 1) + temp.z*rotationMatrix(2, 2);
	p = p + bias;

	return p;
}
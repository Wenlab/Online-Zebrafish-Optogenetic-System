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

void zbb2FishImg::initialize(std::string filename)
{
	string zbbPath = filename;
	vector<pair<string, Point>> zbbMapPair;
	zbbMapPair = readZBBMapFromTxt(zbbPath);
	zbbMapVec = makeZbbMapVec(zbbMapPair);
}

void zbb2FishImg::getRegionFromUser(cv::Rect Reg)
{
	Region = Reg;

	//Converting regions to coordinates
	bool inverse = false;
	for (int i = Region.x; i < Region.x + Region.width; i++)
	{
		if (!inverse)
		{
			for (int j = Region.y; j < Region.y + Region.height; j++)
			{
				Point3f temp{ (float)i,(float)j,0 };
				RegionCoord.push_back(temp);
			}
		}
		if (inverse)
		{
			for (int j = Region.y + Region.height; j > Region.y; j--)
			{
				Point3f temp{ (float)i,(float)j,0 };
				RegionCoord.push_back(temp);
			}
		}

		inverse = !inverse;
	}

	//Query the brain areas contained in this region
	BrainRegionName = queryRegionName(Region);

	return;
}

//Get the affine matrix from ZBB to Fix Image
void zbb2FishImg::getZBB2FixAffineMatrix(std::vector<float> Fix2zbbAM)
{
	zbb2FixAM = inverseAffineMatrix(Fix2zbbAM);
	return;
}

//Get the affine matrix from fix Image to moving Image
void zbb2FishImg::getFix2MovingAffineMatrix(std::vector<float> Moving2FixAM)
{
	Fix2MovingAM = inverseAffineMatrix(Moving2FixAM);
	return;
}

//Get the crop information from the rotated image to the moving image
void zbb2FishImg::getCropPoint(cv::Point3f pt)
{
	cropPoint = pt;
}

//Get the rotation angle from the original image to the rotated image
//Rotation around Z-axis, rotation around X-axis
void zbb2FishImg::getRotationMatrix(float rotationAngleZ, float rotationAngleX)
{
	//Rotation matrix around Z-axis
	rotationAngleZ = rotationAngleZ * PI / 180.0;
	Eigen::Matrix3f rotationMatrixZ;
	rotationMatrixZ << cos(rotationAngleZ), -sin(rotationAngleZ), 0,
		sin(rotationAngleZ), cos(rotationAngleZ), 0,
		0, 0, 1;
	//Rotation matrix around the X-axis
	rotationAngleX = rotationAngleX * PI / 180.0;
	Eigen::Matrix3f rotationMatrixX;
	rotationMatrixX << 1, 0, 0,
		0, cos(rotationAngleX), -sin(rotationAngleX),
		0, sin(rotationAngleX), cos(rotationAngleX);

	//Multiply two rotation matrices together to find the total rotation matrix
	rotationMatrix = rotationMatrixZ;
	return;
}


//Convert area from selected coordinates on ZBB to fish
std::vector<cv::Point2f> zbb2FishImg::ZBB2FishTransform()
{
	for (int i = 0; i < RegionCoord.size(); i++)
	{
		Point3f p;
		Point3f temp = RegionCoord[i];
		p = applyAffineMatrixOn3DCoord(temp, zbb2FixAM);
		p = applyAffineMatrixOn3DCoord(p, Fix2MovingAM);
		p = p + cropPoint;
		p = applyRotationMatrixOn3DCoord(p, rotationMatrix);


		regionInFish.push_back(cv::Point(p.x, p.y));
	}

	return regionInFish;
}


vector<pair<string, Point>> zbb2FishImg::readZBBMapFromTxt(string file)
{
	vector<pair<string, Point>> zbbMap;

	ifstream infile;
	infile.open(file.data()); 
	assert(infile.is_open()); 

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
	infile.close();         
	return zbbMap;
}

vector<vector<vector<string>>> zbb2FishImg::makeZbbMapVec(vector<pair<string, Point>> zbbMapPair)
{
	int a = 77;
	int b = 95;
	vector<vector<vector<string>>> zbbMapVec(a, vector<vector<string>>(b));

	for (int i = 0; i < zbbMapPair.size(); i++)
	{
		pair<string, Point> temp = zbbMapPair[i];
		//The coordinates saved by matlab start from 1
		zbbMapVec[temp.second.x - 1][temp.second.y - 1].push_back(temp.first);   
	}
	//Remove the same element at each pixel position
	for (int i = 0; i < zbbMapVec.size(); i++)
	{
		for (int j = 0; j < zbbMapVec[i].size(); j++)
		{
			zbbMapVec[i][j].erase(unique(zbbMapVec[i][j].begin(), zbbMapVec[i][j].end()), zbbMapVec[i][j].end());
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
			for (int m = 0; m < pixelLable.size(); m++)
			{
				queryRegion.push_back(pixelLable[m]);
			}
		}
	}

	//Remove duplicate region names
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

	Eigen::Matrix3f am_eigen; 
	for (int a = 0; a < 3; a++)
	{
		for (int b = 0; b < 3; b++)
		{
			am_eigen(b, a) = am[a * 3 + b];
		}
	}
	Eigen::Matrix3f am_eigen_inverse = am_eigen.inverse(); 

	Eigen::Vector3f am_trans;
	am_trans << am[9], am[10], am[11];
	Eigen::Vector3f am_trans_inverse = am_trans.transpose() * (-am_eigen_inverse);


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
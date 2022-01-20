#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#pragma warning(disable : 4996)

vector<pair<string, Point>> readZBBMapFromTxt(string file);

vector<vector<vector<string>>> makeZbbMapVec(vector<pair<string, Point>> zbbMapPair);

vector<string> queryRegionName(Rect region, vector<vector<vector<string>>> zbbMapVec);

//bool cmp(string a, string b)
//{
//	return strcmp(a.c_str(), b.c_str());
//}

int main()
{
	string zbbPath = "anatomyList_4bin.txt";

	vector<pair<string, Point>> zbbMapPair;
	zbbMapPair = readZBBMapFromTxt(zbbPath);
	vector<vector<vector<string>>> zbbMapVec = makeZbbMapVec(zbbMapPair);


	//ѡ�����λ��
	Rect lightArea(50, 50, 10, 10);
	vector<string> regionName = queryRegionName(lightArea, zbbMapVec);
	//check
	for (int i = 0; i < regionName.size(); i++)
	{
		cout << regionName[i] << endl;
	}
	vector<Point3f> RegionCoord;   //zbb��ֱ��ѡ�����������Ƕ�ά��Ϊ��3D������չ����ά
	for (int i = lightArea.x; i < lightArea.x + lightArea.width; i++)
	{
		for (int j = lightArea.y; j < lightArea.y + lightArea.height; j++)
		{
			Point3f temp{ (float)i,(float)j,0 };
			RegionCoord.push_back(temp);
			//cout << temp << endl;
		}
	}

	//��zbb��fixImage�������任
	vector<float> zbb2FixAM{ 0.995751 ,-0.0000857003 ,0.00319121 ,
		-0.000161462 ,1.06928 ,0.000254662 ,
		0.0136118 ,-0.0196541 ,1.22397 ,
		0.0194115 ,-11.6822 ,33.888 };


	//��fixImage��MovingImage�������任

	//��MovingImage(77*95*52)��ԭͼ(200*200*50)


	Mat zbbMIP = imread("Ref-zbb-MIP.png");
	rectangle(zbbMIP, lightArea, Scalar(255), 1);

	namedWindow("zbbMIP", 0);
	resizeWindow("zbbMIP", Size(zbbMIP.cols * 4, zbbMIP.rows * 4));

	imshow("zbbMIP", zbbMIP);
	waitKey(0);

	return 0;
}


vector<pair<string, Point>> readZBBMapFromTxt(string file)
{
	vector<pair<string, Point>> zbbMap;

	ifstream infile;
	infile.open(file.data());   //���ļ����������ļ��������� 
	assert(infile.is_open());   //��ʧ��,�����������Ϣ,����ֹ�������� 

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
	infile.close();             //�ر��ļ������� 

	return zbbMap;
}

vector<vector<vector<string>>> makeZbbMapVec(vector<pair<string, Point>> zbbMapPair)
{
	int a = 77;
	int b = 95;
	vector<vector<vector<string>>> zbbMapVec(a, vector<vector<string>>(b));

	cout << zbbMapPair.size() << endl;
	for (int i = 0; i < zbbMapPair.size(); i++)
	{
		pair<string, Point> temp = zbbMapPair[i];
		zbbMapVec[temp.second.x - 1][temp.second.y - 1].push_back(temp.first);   //matlab����������Ǵ�1��ʼ
	}
	//ȥ��ÿ������λ�õ���ͬԪ��
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

vector<string> queryRegionName(Rect region, vector<vector<vector<string>>> zbbMapVec)
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

	//ɾ���ظ�����������
	sort(queryRegion.begin(), queryRegion.end());
	queryRegion.erase(unique(queryRegion.begin(), queryRegion.end()), queryRegion.end());

	return queryRegion;
}

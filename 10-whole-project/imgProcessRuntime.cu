//#define _CRT_SECURE_NO_WARNINGS
//
//#include"imgProcess.h"
////#include "imgProcess.cu"
//#include"kexinLibs.h"
//
//#include<vector>
//#include<iostream>
//
//#include"Timer.h"
//
//using std::string;
//using std::vector;
//using std::cout;
//using std::endl;
//
//
//
//int main()
//{
//	//计时
//	Timer timer1;
//	
//
//
//	//读取PSF和未重构的文件
//	string PSF_1_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/PSF_1_zhuanzhi_float.dat";//matlab中保存出来的float类型
//	string X31_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/r20210924_2_X31_resize.tif";
//	//读取角度、二维模板信息
//	string rotationAngleXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleXY.dat";//360个double
//	string rotationAngleYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleYZ.dat";//31个double
//	string template_roXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/templateXY.tif";//200*200*360个float，按照matlab中行优先存储，存完一个波段再存第二个波段
//	string template_roYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/template_roYZ.dat";//200*50*31个float，按照matlab中行优先存储，存完一个波段再存第二个波段
//	//读取用于仿射对齐的fixImage
//	string fixImage_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/toAffineWithZBB.tif";
//
//
//	string imgBeforeRecon_path = "D:/kexin/Online-Zebrafish-Optogenetic/data/r20210824_X10_4bin";
//	vector<string> imgBeforeRecon_Names;
//	getFileNames(imgBeforeRecon_path, imgBeforeRecon_Names);
//
//	string modelPath = "affineNetScript_TM_0621_3080.pt";
//
//	FishImageProcess fishImgProc(modelPath);
//	
//	fishImgProc.readPSFfromFile(PSF_1_file);
//	fishImgProc.readRotationAngleFromFile(rotationAngleXY_file, rotationAngleYZ_file);
//	fishImgProc.readTemplateFromFile(template_roXY_file, template_roYZ_file);
//	fishImgProc.readFixImageFromFile(fixImage_file);
//	                                                                                                                                   
//	fishImgProc.initializeFishReg("anatomyList_4bin.txt");
//
//	fishImgProc.prepareGPUMemory();
//	fishImgProc.processPSF();
//
//	for (int i = 0; i < imgBeforeRecon_Names.size(); i++)
//	{
//		fishImgProc.readImageFromFile(imgBeforeRecon_Names[i]);
//
//		timer1.start();
//		fishImgProc.reconImage();//重构读进来的图像
//		fishImgProc.cropReconImage();   
//
//		//rotation
//		fishImgProc.matchingANDrotationXY();
//
//		//crop
//		fishImgProc.cropRotatedImage();
//
//		////crop的结果构建movingTensor，和fixTensor一起送入网络处理
//		fishImgProc.libtorchModelProcess();
//
//		////结合rotation/crop/affine的数据做坐标转换
//		std::vector<cv::Point3f> points = fishImgProc.ZBB2FishTransform();
//
//
//		timer1.stop();
//		cout << "time cost: " << timer1.getElapsedTimeInMilliSec() << " ms" << endl;
//
//		//for (int j = 0; j < points.size(); j++)
//		//{
//		//	cout << points[j] << endl;
//		//}
//
//		//save and check
//		string saveName1 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testRecon/" + int2string(4, i) + ".tif";
//		string saveName2 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testMatchingXY/" + int2string(4, i) + ".tif";
//		string saveName3 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testCrop/" + int2string(4, i) + ".tif";
//		
//		//76 * 95 * 50
//		//200*200*50
//		//test reconstruction
//		float* temp = new float[200 * 200 * 50]();
//		cudaMemcpy(temp, fishImgProc.gpuObjRecon_crop, sizeof(float) * 200 * 200 * 50, cudaMemcpyDeviceToHost);
//		saveAndCheckImage(temp, 200, 200, 50, saveName1);
//		//test rotation
//		float* temp1 = new float[200 * 200 * 50]();
//		cudaMemcpy(temp1, fishImgProc.imageRotated3D_gpu, sizeof(float) * 200 * 200 * 50, cudaMemcpyDeviceToHost);
//		saveAndCheckImage(temp1, 200 , 200 , 50, saveName2);
//		//test crop
//		float* temp2 = new float[76 * 95 * 50]();
//		cudaMemcpy(temp2, fishImgProc.ObjCropRed_gpu, sizeof(float) * 76 * 95 * 50, cudaMemcpyDeviceToHost);
//		saveAndCheckImage(temp2, 76 , 95 , 49, saveName3);
//
//		free(temp);
//		free(temp1);
//		free(temp2);
//	}
//
//	fishImgProc.freeMemory();
//
//	return 0;
//}
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
//	//��ʱ
//	Timer timer1;
//	
//
//
//	//��ȡPSF��δ�ع����ļ�
//	string PSF_1_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/PSF_1_zhuanzhi_float.dat";//matlab�б��������float����
//	string X31_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/r20210924_2_X31_resize.tif";
//	//��ȡ�Ƕȡ���άģ����Ϣ
//	string rotationAngleXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleXY.dat";//360��double
//	string rotationAngleYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/old/rotationAngleYZ.dat";//31��double
//	string template_roXY_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/templateXY.tif";//200*200*360��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
//	string template_roYZ_file = "D:/kexin/Online-Zebrafish-Optogenetic/data/template_roYZ.dat";//200*50*31��float������matlab�������ȴ洢������һ�������ٴ�ڶ�������
//	//��ȡ���ڷ�������fixImage
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
//		fishImgProc.reconImage();//�ع���������ͼ��
//		fishImgProc.cropReconImage();   
//
//		//rotation
//		fishImgProc.matchingANDrotationXY();
//
//		//crop
//		fishImgProc.cropRotatedImage();
//
//		////crop�Ľ������movingTensor����fixTensorһ���������紦��
//		fishImgProc.libtorchModelProcess();
//
//		////���rotation/crop/affine������������ת��
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
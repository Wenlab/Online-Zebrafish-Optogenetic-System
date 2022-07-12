#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS
#define PROCESSOR_3_MASK 0x01<<3

#include <chrono>
#include<thread>
#include<functional>
#include<windows.h>
#include<timeapi.h>
#pragma comment(lib,"WinMM.Lib")

#include"experiment.h"
#include"kexinLibs.h"

#include <direct.h> //_mkdir fun
#include "main.h"


using namespace std;

int main()
{
	
	string afterResizeImgPath = "D:/kexin/Online-Zebrafish-Optogenetic/data/r20210824_X10_resize/";
	_mkdir(afterResizeImgPath.data());

	Experiment myExp("affineNetScript_TM_0621_3080.pt");

	myExp.initialize();


	//thread cameraGrabLoopThread(&Experiment::readFullSizeImgFromFile, &myExp);   //文件接口
	thread cameraGrabLoopThread(&Experiment::readFullSizeImgFromCamera, &myExp);   //相机接口
	thread imgProcessLoopThread(&Experiment::imgProcess, &myExp);
	thread writeOutLoopThread(&Experiment::writeOutData, &myExp);
	thread galvoControlThread(&Experiment::galvoControl, &myExp);


	myExp.controlExp();


	cameraGrabLoopThread.join();
	imgProcessLoopThread.join();
	writeOutLoopThread.join();
	galvoControlThread.join();

	myExp.clear();

	//myExp.exit();

	//for (int i = 0; i < beforeResizeImgNames.size(); i++)
	//{
	//	myExp.getParamsFromGUI();

	//	if (myExp.UserWantToStop)
	//		break;

	//	myExp.readFullSizeImgFromFile(beforeResizeImgNames[i]);
	//	myExp.resizeImg();

	//	myExp.ImgReconAndRegis();

	//	myExp.getReconMIP();

	//	//cv::imshow("test",myExp.MIP);
	//	myExp.drawGUIimg();

	//	//test write out
	//	myExp.writeOutData();

	//	////save and check
	//	//string saveName1 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testRecon/" + int2string(4, i) + ".tif";
	//	//string saveName2 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testMatchingXY/" + int2string(4, i) + ".tif";
	//	//string saveName3 = "D:/kexin/Online-Zebrafish-Optogenetic/data/testCrop/" + int2string(4, i) + ".tif";

	//	////76 * 95 * 50
	//	////200*200*50
	//	////test reconstruction
	//	//float* temp = new float[200 * 200 * 50]();
	//	//cudaMemcpy(temp, myExp.fishImgProc.gpuObjRecon_crop, sizeof(float) * 200 * 200 * 50, cudaMemcpyDeviceToHost);
	//	//saveAndCheckImage(temp, 200, 200, 50, saveName1);
	//	////test rotation
	//	//float* temp1 = new float[200 * 200 * 50]();
	//	//cudaMemcpy(temp1, myExp.fishImgProc.imageRotated3D_gpu, sizeof(float) * 200 * 200 * 50, cudaMemcpyDeviceToHost);
	//	//saveAndCheckImage(temp1, 200, 200, 50, saveName2);
	//	////test crop
	//	//float* temp2 = new float[76 * 95 * 50]();
	//	//cudaMemcpy(temp2, myExp.fishImgProc.ObjCropRed_gpu, sizeof(float) * 76 * 95 * 50, cudaMemcpyDeviceToHost);
	//	//saveAndCheckImage(temp2, 76, 95, 49, saveName3);

	//	//free(temp);
	//	//free(temp1);
	//	//free(temp2);
	//}
	

	return 0;

}
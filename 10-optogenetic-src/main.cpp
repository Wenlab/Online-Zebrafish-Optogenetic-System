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

#include <direct.h>



using namespace std;

int main()
{

	Experiment myExp("affineNetScript_TM_0621_3080.pt");

	myExp.initialize();

	thread galvoControlThread(&Experiment::galvoControl, &myExp);
	SetThreadPriority(galvoControlThread.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
	SetThreadAffinityMask(galvoControlThread.native_handle(), PROCESSOR_3_MASK);

	thread cameraGrabLoopThread(&Experiment::readFullSizeImgFromCamera, &myExp);  
	thread imgProcessLoopThread(&Experiment::imgProcess, &myExp);
	thread writeOutLoopThread(&Experiment::writeOutData, &myExp);
	thread TCPconnectLoopThread(&Experiment::TCPconnect, &myExp);



	myExp.controlExp();


	galvoControlThread.join();
	cameraGrabLoopThread.join();
	imgProcessLoopThread.join();
	writeOutLoopThread.join();
	TCPconnectLoopThread.join();

	myExp.clear();

	

	return 0;

}
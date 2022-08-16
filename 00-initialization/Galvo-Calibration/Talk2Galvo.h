#pragma once
#pragma once

#ifndef TALK2GALVO_H_
#define TALK2GALVO_H_

#include <windows.h>
#include "NIDAQmx.h"
#include "opencv2/opencv.hpp"

class GalvoData
{
private:
	;
public:
	// methord
	GalvoData()
	{
		taskHandle1 = NULL;
		taskHandle2 = NULL;
	}

	void handleDAQError(int error);
	bool initialize();
	int spinGalvo(cv::Point2f pt);


	TaskHandle taskHandle1, taskHandle2;
	//TaskHandle taskHandle2;
};


#endif /* TALK2GALVO_H_ */
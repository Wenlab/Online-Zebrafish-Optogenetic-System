#include <stdio.h>
#include <math.h>
#include <windows.h>

#include "Talk2Galvo.h"


void GalvoData::handleDAQError(int error)
{
	char  errBuff[2048] = { '\0' };

	if (DAQmxFailed(error))
		DAQmxGetExtendedErrorInfo(errBuff, 2048);
	if (taskHandle1 != 0) 
	{
		/*********************************************/
		// DAQmx Stop Code
		/*********************************************/
		DAQmxStopTask(taskHandle1);
		DAQmxClearTask(taskHandle1);
	}
	if (DAQmxFailed(error))
		printf("DAQmx Error: %s\n", errBuff);
	printf("End of program, press Enter key to quit\n");
	getchar();
}

bool GalvoData::initialize()
{
	int error = 0;

	error = DAQmxCreateTask("", &taskHandle1);

	if (error != 0) 
	{
		handleDAQError(error);
		printf("ERROR! Cannot control the stage with DAQ card.\n");
		taskHandle1 = NULL;
		return false;
	}


	error = DAQmxCreateAOVoltageChan(taskHandle1, "Dev3/ao0:1", "", -5.0, 5.0, DAQmx_Val_Volts, "");

	if (error != 0)
	{
		handleDAQError(error);
		printf("ERROR! Cannot control the stage with DAQ card.\n");
		taskHandle1 = NULL;
		return false;
	}


	//第一个值尽量大，第二个值尽量小
	error = DAQmxCfgSampClkTiming(taskHandle1, "", 50000, DAQmx_Val_Rising, DAQmx_Val_ContSamps, 5000);

	if (error != 0)
	{
		handleDAQError(error);
		printf("ERROR! Cannot control the stage with DAQ card.\n");
		taskHandle1 = NULL;
		return false;
	}

	error = DAQmxStartTask(taskHandle1);
	if (error != 0)
	{
		handleDAQError(error);
		printf("ERROR! Invoking the stage failed.\n");
		return false;
	}

	return true;
}


int GalvoData::spinGalvo(cv::Point2f pt)
{
	int32 written;
	int	error = 0;


	double data[2];
	data[0] = pt.x;
	data[1] = pt.y;

	error = DAQmxWriteAnalogF64(taskHandle1, 1, 0, 10.0, DAQmx_Val_GroupByChannel, data, &written, NULL);

	if (error != 0) 
	{
		handleDAQError(error);
		printf("ERROR! cannot send speed control commands.\n");
		return -1;
	}

	return 0;
}
/*
 * Talk2Camera.h
 *
 *  Created on: Sep 3, 2013
 *      Author: quan Wen, modified from Andy's code
 */


#pragma once
#ifndef TALK2CAMERA_H_
#define TALK2CAMERA_H_

#include <atcore.h>
#include <stdio.h>

#define PRINT_DEBUG 0
#define NumberOfBuffers 20 //Number of Buffer used to get the image, if GPuExpPath is enabled, should have NumberOfBuffers >= NumberOfPaths*BuffersPerPath
#define NumberOfPaths 4 //Number of GpuExpPath
#define BuffersPerPath 1 //Number of images stored in each Path

/*
 * We define a new variable type, "CamData" which is a struct that
 * holds information about the current Data in the camera.
 *
 * The actual data resides in *iImageData
 * The i notation indicates that these are internal values. e.g.
 * iFrameNumber refers to the FrameNumber that the camera sees,
 *
 */
typedef struct CamDataStruct CamData;

struct CamDataStruct {


	AT_64 ImageHeight; 
	AT_64 ImageWidth; 
	AT_64 ImageStride; 
	AT_64 ImageSizeBytes;
	AT_WC PixelEncoding[64]; 
	unsigned long long iFrameNumber;
	unsigned short* ImageRawData;
	unsigned char* AcqBuffers[NumberOfBuffers];
	unsigned char* AlignedBuffers[NumberOfBuffers];
	};

/*
 * Rows and pixels of camera
 */
#define CCDSIZEX 2048
#define CCDSIZEY 2048
#define CCDENCODING L"Mono16" //L"Mono12Packed" or L"Mono12" or L"Mono16"

/*
 * Initalizes the library and provides the  license key for
 * the Imaging control software. The function returns a
 * non-zero value if successful.
 */
int T2Cam_InitializeLib(int* Hndl);

/*
 * Closes the library.
 *
 */
void T2Cam_CloseLib();



/*
 * Create CamData type, this function will allocate
 * memory for raw image data.
 */

CamData* T2Cam_CreateCamData();

void T2Cam_InitializeCamData(CamData* MyCamera,int _handle);


int T2Cam_GrabFrame(CamData* MyCamera, int _handle);

void T2Cam_TurnOff(CamData* MyCamera,int _handle);

void SetupSensorCooling(int _handle);

int CreateBuffer(CamData* MyCamera,int _handle);

void T2Cam_StartAcquisition(int _handle);

int getUserSettings(int _handle);

int SetupBinningandAOI(int _handle);

void deleteBuffers(CamData* MyCamera);


int AutogetUserSettings(int _handle);


void T2Cam_Close(CamData* MyCamera, AT_H _handle);


#endif /* TALK2CAMERA_H_ */

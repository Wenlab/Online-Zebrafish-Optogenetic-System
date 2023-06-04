/*
 * Copyright 2010 Andrew Leifer et al <leifer@fas.harvard.edu>
 * This file is part of MindControl.
 *
 * MindControl is free software: you can redistribute it and/or modify
 * it under the terms of the GNU  General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MindControl is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MindControl. If not, see <http://www.gnu.org/licenses/>.
 *
 * For the most up to date version of this software, see:
 * http://github.com/samuellab/mindcontrol
 *
 *
 *
 * NOTE: If you use any portion of this code in your research, kindly cite:
 * Leifer, A.M., Fang-Yen, C., Gershow, M., Alkema, M., and Samuel A. D.T.,
 * 	"Optogenetic manipulation of neural activity with high spatial resolution in
 *	freely moving Caenorhabditis elegans," Nature Methods, Submitted (2010).
 */

/*
 * Talk2Camera.c
 *
 * Created on 27 July 2009
 *   by Anddrew Leifer
 *   leifer@fas.harvard.edu
 *
 * Modified on 27 Februrary 2016	
 *   by Quan Wen
 *	 qwen@ustc.edu.cn
 *
 *
 *   Talk2Camera is a library to interface with Andor sCMOScamera's
 *   It uses a number of supplied wrapper libraries from the Andor sCMOS
 *
 *   It depends on: atcore.h, atcorem.lib and atcore.dll
 *
 */



#include <atcore.h>
#include <atutility.h>
#include "Talk2Camera.h"
#include <stdio.h>
#include <stdlib.h>


#include <iostream> 
using namespace std;



// Do we print debugging output?


/*
 * Initalizes the library and provides the  license key for
 * the Imaging control software. The function returns a
 * non-zero value if successful.
 */
int T2Cam_InitializeLib(int* Hndl) {
	//HardCode in the key to initialize the Andor sCMOS Library
	int i_retCode;
  	
  	cout << "Initialising ..." << endl << endl;
  	
  	i_retCode = AT_InitialiseLibrary();
  	
  	if (i_retCode != AT_SUCCESS) {
    	cout << "Error initialising library" << endl << endl;
  	}
  	
  	else {

  		i_retCode = AT_InitialiseUtilityLibrary();

  		if (i_retCode == AT_SUCCESS){

            //Check the number of cameras then choose the camera to turn on
  			AT_64 iNumberDevices = 0;
    		AT_GetInt(AT_HANDLE_SYSTEM, L"DeviceCount", &iNumberDevices);
    	    cout << iNumberDevices << " cameras detected. " << endl;
            
            if (iNumberDevices <= 0) 
			{
      			cout << "No cameras detected"<<endl;
    		}
    	
    		else {
      				
      			i_retCode = AT_Open(0, Hndl);

      			if (i_retCode != AT_SUCCESS) 
				{
        			cout << "Error condition, could not initialise camera" << endl << endl;
      			}

      			else {
        		
        			cout << "Successfully initialised camera_0" << endl << endl;

        			AT_WC szValue[64];
        			i_retCode= AT_GetString(*Hndl, L"Serial Number", szValue, 64);

        			if (i_retCode == AT_SUCCESS) {
          				wcout << L"The serial number is " << szValue << endl;
        			}
        		
        			else {

          				cout << "Error obtaining Serial number" << endl << endl;
        			}

                    //Set encoding mode
  					i_retCode = AT_SetEnumeratedString(*Hndl, L"PixelEncoding", CCDENCODING );
        			if (i_retCode == AT_SUCCESS) {
          				wcout << "Pixel Encoding set to " << CCDENCODING << endl << endl;
        			}
                    else{
                        wcout<<"Cannot Set Pixel Encoding to " << CCDENCODING <<endl<<endl;
                    }
        			

        			int iret = AT_SetEnumeratedString(*Hndl, L"TriggerMode",L"Internal");
        			if (iret != AT_SUCCESS ) {
          				cout << "Error setting trigger mode to Internal, retcode=" << iret << endl;
        			}
        			else {
          				cout << "Trigger mode set to Internal" << endl << endl;
        			}

        			iret = AT_SetEnumeratedString(*Hndl, L"CycleMode", L"Continuous");
        		
        			if (iret != AT_SUCCESS ) {
          				cout << "Error setting Cycle Mode to Continuous, retcode=" << iret << endl;
        			}
        
        			else {

          				cout << "CycleMode set to Continuous" << endl << endl;
        			}

					//cout << "cooling..." << endl;
					//SetupSensorCooling(*Hndl);

					//set simple preAmpGainControl
					//set sensitivity/dynamic range to 16-bit (low noise & high well capacity)

					iret = AT_SetEnumeratedString(*Hndl, L"SimplePreAmpGainControl", L"16-bit (low noise & high well capacity)");
					if (iret != AT_SUCCESS) {
						cout << "Error setting SimplePreAmpGainControl to 16-bit (low noise & high well capacity), retcode=" << iret << endl;
					}

					else {

						cout << " setting SimplePreAmpGainControl to 16-bit (low noise & high well capacity)" << endl << endl;
					}

        		}

  			}

    	}

    }

    return i_retCode;

}

void T2Cam_CloseLib() {
	
  	AT_FinaliseLibrary();
    AT_FinaliseUtilityLibrary();
    printf("Andor Library closed! \n");
}

void T2Cam_TurnOff(CamData* MyCamera,int _handle) {

  printf("Inside T2Cam_TurnOff()\n");
  delete[] MyCamera->ImageRawData;
  deleteBuffers(MyCamera);
  free(MyCamera);
  AT_Command(_handle, L"AcquisitionStop");
  AT_Flush(_handle);    
  AT_Close(_handle);
  printf("Exiting T2Cam_TurnOff()\n");

}



CamData* T2Cam_CreateCamData(){

	CamData* MyCamera = (CamData*) malloc(sizeof(CamData));
	MyCamera->ImageSizeBytes = 0;
	MyCamera->ImageHeight = 0;
	MyCamera->ImageWidth = 0;
	MyCamera->ImageStride = 0; 
	MyCamera->ImageRawData = NULL;
	MyCamera->iFrameNumber = 0;

	for (int i=0; i < NumberOfBuffers; i++) { 

		MyCamera->AcqBuffers[i] = NULL;

		MyCamera->AlignedBuffers[i] = NULL;

		//MyCamera->ResizeBuffers[i] = NULL;

	} 


 	return MyCamera; 
}


void T2Cam_InitializeCamData(CamData* MyCamera,int _handle) 
{
	printf("inside InitializeCamData\n");
	AT_GetInt(_handle, L"ImageSizeBytes", &(MyCamera->ImageSizeBytes));
	AT_GetInt(_handle, L"AOIHeight", &(MyCamera->ImageHeight));
	AT_GetInt(_handle, L"AOIWidth", &(MyCamera->ImageWidth));
	AT_GetInt(_handle, L"AOIStride", &(MyCamera->ImageStride));
	AT_GetString(_handle,L"PixelEncoding", MyCamera->PixelEncoding, 64);
	MyCamera->ImageRawData = new unsigned short[static_cast<size_t>(MyCamera->ImageHeight*MyCamera->ImageWidth)];

}


/*
 * Set's up the camera in continuous mode and tells the camera to begin
 * continuously grabbing frames and calling the callback function.
 *  This requires a pointer to a CamData type
 * that has memory already allocated.
 *
 * This begins thread that calls a Callback function
 *  that immediately begins dumping data into CameraDataStruct
 */
int T2Cam_GrabFrame(CamData* MyCamera, int _handle) {
	
    //AT_U8* pBuf;
    unsigned char* pBuf;
	int BufSize = static_cast<int>(MyCamera->ImageSizeBytes);
	//int BufSize = static_cast<int>(MyCamera->ImageSizeBytes);
    
    
    //issue software trigger command

    //iret = AT_Command(_handle, L"SoftwareTrigger");
    //if (iret!=AT_SUCCESS){
    //  cout << "Error:Return from Software trigger command not success " << iret << endl;
    //  return 1;
    //}
    //now do a wait and if its not success then error

	int iret;
	int iError;
  //int data[2];

    /*unsigned char** ppBuf=&pBuf;
    int* pBufSize = &BufSize;*/
    iret = AT_WaitBuffer(_handle, &pBuf, &BufSize, AT_INFINITE);   //AT_INFINITE  Wait until the queue has data
    if (iret!=AT_SUCCESS){
      cout << "Error:Acquisition timeout when not expecting, retcode " << iret << endl;
      return -1;
    }

    
    iError = AT_ConvertBuffer(pBuf, reinterpret_cast<unsigned char*>(MyCamera->ImageRawData), \
        MyCamera->ImageWidth, MyCamera->ImageHeight, MyCamera->ImageStride, CCDENCODING , L"Mono16");
    if (iError != AT_SUCCESS) {
        cout << "Convert image format failed- return code " << iError << endl;
        return -1;
    }

    //cout << "    First 2 pixels " << MyCamera->ImageRawData[0] << " " << MyCamera->ImageRawData[1]<< endl;



    //printf("grabbed a frame! \n");

    //extract2from3(pBuf,data);

    //cout << "    First 2 pixels " << data[1] << " " << data[0] << endl;

    //iError = AT_QueueBuffer(_handle, MyCamera->AlignedBuffers[MyCamera->FrameNumber%NumberOfBuffers], static_cast<int>(MyCamera->ImageSizeBytes));
    iError = AT_QueueBuffer(_handle, pBuf, BufSize);
    

    if (iError != AT_SUCCESS) {
    	  cout << "AT_QueueBuffer failed - Image Size Bytes - return code " << iError << endl;
    }
    MyCamera->iFrameNumber++;
    //cout << "Got image " << MyCamera->FrameNumber  << endl;

    

    return 0;
}

/*
 *
 * Takes a CameraDataStruct with an initialized frame grabber
 * and turns it off. De Allocates Camera Data.
 *
 */

void T2Cam_StartAcquisition(int _handle){

	AT_Command(_handle, L"AcquisitionStart");
  //Sleep(100);

}





void deleteBuffers(CamData* MyCamera){

  for (int i=0; i < NumberOfBuffers; i++) { delete[] MyCamera->AcqBuffers[i]; } 


}


int CreateBuffer(CamData* MyCamera, int _handle) 
{

  int iError;
  // Get the number of bytes required to store one frame


  //AT_64 ImageSizeBytes = 0;
  //iError = AT_GetInt(_handle, L"ImageSizeBytes", &ImageSizeBytes);
  //if (iError != AT_SUCCESS) {
  //  cout << "AT_GetInt failed - ImageSizeBytes - return code " << iError << endl;
  //}



  cout << "ImageWidth is " << MyCamera->ImageWidth << endl;
  cout << "ImageHeight is " << MyCamera->ImageHeight << endl;
  cout << "ImageSizeBytes is " << MyCamera->ImageSizeBytes << endl;

  wcout<< " Pixel encoding is " << MyCamera->PixelEncoding <<endl;
    

  	for (int i=0; i < NumberOfBuffers; i++){
    	// Allocate a memory buffer to store one frame

    	//MyCamera->AcqBuffers[i]=(unsigned char *) malloc((ImageSizeBytes+7) * sizeof(unsigned char));

    	MyCamera->AcqBuffers[i] = new unsigned char[MyCamera->ImageSizeBytes + 7];
    	//printf("allocate buffers \n");
    	//MyCamera->AlignedBuffers[i] = MyCamera->AcqBuffers[i];//Buffers can not be aligened right by the way below (because unsigned long is not "long" enough), so I just copy AcqBuffers.
        //Make sure the data is 64-bit aligned, necessary for bitflow SDK
        MyCamera->AlignedBuffers[i] = reinterpret_cast<unsigned char*>((reinterpret_cast<unsigned long long>(MyCamera->AcqBuffers[i%NumberOfBuffers]) + 7) & ~7);


    }



    for (int i=0; i < NumberOfBuffers; i++){

    	// Pass this buffer to the SDK
    	iError = AT_QueueBuffer(_handle, MyCamera->AlignedBuffers[i], MyCamera->ImageSizeBytes);

	    if (iError != AT_SUCCESS) {
    	  cout << "AT_QueueBuffer failed - Image Size Bytes - return code " << iError << endl;
    	}


    }

 
 

  return iError;
 
}

int getUserSettings(int _handle){
	int i_retCode;
	//set trigger mode   
	// hardware or software
	int i_trigger;
	cout << endl << "Enter trigger mode, 1 for hardware, 0 or else for software" << endl;
	cin >> i_trigger;
	if (i_trigger == 1)
	{
		i_retCode= AT_SetEnumString(_handle,L"TriggerMode", L"External Exposure");
	}
	else
	{
		i_retCode = AT_SetEnumString(_handle, L"TriggerMode", L"Internal");
	}
	if (i_retCode != AT_SUCCESS)
	{
		cout << "Error setting Trigger Mode " << endl;
	}


    //Set Readout Rate
    cout << endl << "Enter the pixel Readout rate, 100, 200, 270 or 540." << endl;
    int i_rate = 270;
    //cin >> i_rate;
	cout << "set pixel readout rate to: " << i_rate << endl;
    if (i_rate == 100) {
        i_retCode = AT_SetEnumIndex(_handle, L"PixelReadoutRate", 1);
    }
    else if (i_rate == 200) {
        i_retCode = AT_SetEnumIndex(_handle, L"PixelReadoutRate", 2);
    }
    else {
       i_retCode = AT_SetEnumIndex(_handle, L"PixelReadoutRate", 3);
     }

    if (i_retCode != AT_SUCCESS) {
    cout << "Error setting Pixel Readout Rate " << i_rate << " MHz" << endl << endl;
    cout << "Error setting Pixel Readout Rate " << i_rate << " MHz" << endl << endl;
    }

    //Check Readout Rate
    int i_index;
    i_retCode = AT_GetEnumIndex(_handle,L"PixelReadoutRate", &i_index);
    if (i_retCode != AT_SUCCESS) {
    cout << "Error getting PixelReadoutRate index " << i_retCode << endl << endl;
    }

    AT_WC szValue[64];
    i_retCode = AT_GetEnumStringByIndex(_handle,L"PixelReadoutRate", i_index, szValue, 64);
    if (i_retCode != AT_SUCCESS) {
    cout << "Error getting PixelReadoutRate string " << i_retCode << endl << endl;
    }
    wcout << "PixelReadoutRate set to " << szValue << endl;

	//Set overlap mode
	AT_BOOL overlapMode = false;
	cout << endl << "Choose the overlap readout mode, 0 or 1." << endl;
	cin >> overlapMode;

	i_retCode = AT_SetBool(_handle, L"Overlap", overlapMode);
	if (i_retCode != AT_SUCCESS) {
		cout << "Error setting Overlap Mode to " << boolalpha << overlapMode << " Error code " << i_retCode << endl << endl;
	}

	//Check overlap mode
	AT_BOOL overlapMode_actual;
	i_retCode = AT_GetBool(_handle, L"Overlap", &overlapMode_actual);
	if (i_retCode != AT_SUCCESS) {
		cout << "Error getting Overlap Mode, Error code " << i_retCode << endl << endl;
	}
	cout << "Overlap Mode set to " << boolalpha << (bool)overlapMode_actual << endl;



	//if software trigger
	if (i_trigger == 0)
	{
		//Set exposure time
		cout << endl << "Enter the Exposure time in seconds, eg 0.009." << endl;
		float f_exp = 0.1;
		//cin >> f_exp;
		cout << "set exposure time to: " << f_exp << endl;

		i_retCode = AT_SetFloat(_handle, L"ExposureTime", f_exp);
		if (i_retCode != AT_SUCCESS) {
			cout << "Error setting Exposure time to " << f_exp << " Error code " << i_retCode << endl << endl;
		}

		//Check exposure time
		double d_actual;
		i_retCode = AT_GetFloat(_handle, L"ExposureTime", &d_actual);
		if (i_retCode != AT_SUCCESS) {
			cout << "Error getting Exposure time, Error code " << i_retCode << endl << endl;
		}
		cout << "Exposure time set to " << d_actual << " second(s)" << endl;

		//Get FrameRate range
		double fps_max = 0;
		double fps_min = 0;
		AT_GetFloatMax(_handle, L"FrameRate", &fps_max);
		AT_GetFloatMin(_handle, L"FrameRate", &fps_min);
		cout << endl << "You can set Frame Rate between " << fps_min << " and " << fps_max << " ." << endl;

		//Set FrameRate
		double fps = 100;
		cout << endl << "Enter the Frame Rate in fps, e.g. 100." << endl;
		cin >> fps;

		i_retCode = AT_SetFloat(_handle, L"FrameRate", fps);
		if (i_retCode != AT_SUCCESS) {
			cout << "Error setting FrameRate to " << fps << " Error code " << i_retCode << endl << endl;
		}

		//Check FrameRate
		double fps_actual;
		i_retCode = AT_GetFloat(_handle, L"FrameRate", &fps_actual);
		if (i_retCode != AT_SUCCESS) {
			cout << "Error getting FrameRate, Error code " << i_retCode << endl << endl;
		}
		cout << "FrameRate set to " << fps_actual << endl;
	}


    return i_retCode;

}

int AutogetUserSettings(int _handle) {
	//Set hardware trigger
	int i_retCode;
	i_retCode = AT_SetEnumString(_handle, L"TriggerMode", L"External Exposure");
	if (i_retCode != AT_SUCCESS)
	{
		cout << "Error setting Trigger Mode " << endl;
	}

    //Set Readout Rate to 270MHz
    i_retCode = AT_SetEnumIndex(_handle, L"PixelReadoutRate", 3);

    if (i_retCode != AT_SUCCESS) {
        cout << "Error setting Pixel Readout Rate 270 MHz" << endl << endl;
    }

    //Check Readout Rate
    int i_index;
    i_retCode = AT_GetEnumerated(_handle, L"PixelReadoutRate", &i_index);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error getting PixelReadoutRate index " << i_retCode << endl << endl;
    }

    AT_WC szValue[64];
    i_retCode = AT_GetEnumeratedString(_handle, L"PixelReadoutRate", i_index, szValue, 64);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error getting PixelReadoutRate string " << i_retCode << endl << endl;
    }
    wcout << "PixelReadoutRate set to " << szValue << endl;

    //Set exposure time to 100ms

    i_retCode = AT_SetFloat(_handle, L"ExposureTime", 0.1);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error setting Exposure time to 9ms." <<  " Error code " << i_retCode << endl << endl;
    }

    //Check exposure time
    double d_actual;
    i_retCode = AT_GetFloat(_handle, L"ExposureTime", &d_actual);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error getting Exposure time, Error code " << i_retCode << endl << endl;
    }
    cout << "Exposure time set to " << d_actual << " second(s)" << endl;

    //Set overlap mode
    i_retCode = AT_SetBool(_handle, L"Overlap", AT_TRUE);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error setting Overlap Mode to " << boolalpha << true << " Error code " << i_retCode << endl << endl;
    }

    //Check overlap mode
    AT_BOOL overlapMode_actual;
    i_retCode = AT_GetBool(_handle, L"Overlap", &overlapMode_actual);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error getting Overlap Mode, Error code " << i_retCode << endl << endl;
    }
    cout << "Overlap Mode set to " << boolalpha << (bool)overlapMode_actual << endl;

    //Set FrameRate

    //i_retCode = AT_SetFloat(_handle, L"FrameRate", 100);
    //if (i_retCode != AT_SUCCESS) {
    //    cout << "Error setting FrameRate to " << 100 << " Error code " << i_retCode << endl << endl;
    //}

    //Check FrameRate
    double fps_actual;
    i_retCode = AT_GetFloat(_handle, L"FrameRate", &fps_actual);
    if (i_retCode != AT_SUCCESS) {
        cout << "Error getting FrameRate, Error code " << i_retCode << endl << endl;
    }
    cout << "FrameRate set to " << fps_actual << endl;

	//check bit depth
	//i_retCode = AT_SetEnumerated(_handle, L"BitDepth", 1);
	//int bitdepth;
	//i_retCode = AT_GetEnumerated(_handle, L"BitDepth", &bitdepth);
	//if (i_retCode != AT_SUCCESS) {
	//	cout << "Error getting bit depth, Error code " << i_retCode << endl << endl;
	//}
	//cout << "bitdepth is " << bitdepth << endl;

    return i_retCode;

}

void SetupSensorCooling(int _handle){

  double temperature = 0;
  AT_SetBool(_handle, L"SensorCooling", AT_TRUE);
  AT_GetFloat(_handle, L"SensorTemperature", &temperature);
  cout << "Temperature: " << temperature << endl;
  int temperatureCount = 0;
  AT_GetEnumCount(_handle, L"TemperatureControl", &temperatureCount);
  AT_SetEnumIndex(_handle, L"TemperatureControl", temperatureCount-1);
  int temperatureStatusIndex = 0;
  wchar_t temperatureStatus[256];
  AT_GetEnumIndex(_handle, L"TemperatureStatus", &temperatureStatusIndex);
  AT_GetEnumStringByIndex(_handle, L"TemperatureStatus", temperatureStatusIndex,
  temperatureStatus, 256);
  
  while(wcscmp(L"Stabilised",temperatureStatus) != 0) {
    Sleep(1);
    AT_GetEnumIndex(_handle, L"TemperatureStatus", &temperatureStatusIndex);
    AT_GetEnumStringByIndex(_handle, L"TemperatureStatus", temperatureStatusIndex,
              temperatureStatus, 256);
    //wcout << L"Temperature Status: " << temperatureStatus << endl;
  }
  cout << "Temperature Stabilised" << endl;
}



int SetupBinningandAOI(int _handle){

    int iret;


    iret = AT_SetEnumString(_handle, L"AOIBinning",L"1x1");


    if (iret != AT_SUCCESS ) {
        cout << "Error setting the Binning to 1x1, retcode=" << iret << endl;
    }

    else {

        cout << "Set Binning to 1x1" << endl;
    }


    AT_64 img_orig_width,width_max; //already take into account of 2x2 binning
    AT_GetIntMax(_handle, L"AOIWidth", & width_max);
    cout << "Max AOI width is " << width_max << endl;
    AT_GetInt(_handle, L"AOIWidth", &img_orig_width);
    iret = AT_SetInt(_handle,L"AOIWidth",CCDSIZEX);
    AT_64 width_actual;
    AT_GetInt(_handle, L"AOIWidth", &width_actual);
    if (iret != AT_SUCCESS ) {
        cout << "Error setting the AOI width to " << CCDSIZEX << ", retcode=" << iret << endl;
    }
    else {
        cout << "Setting the AOI width to " << width_actual << endl;
    }




    AT_64 left_shift;
    //already take into account of the 2x2 binning 
    left_shift=img_orig_width - width_actual +1;
    iret = AT_SetInt(_handle,L"AOILeft",left_shift); 
    if (iret != AT_SUCCESS ) {
        cout << "Error setting the AOI Left shift to " << left_shift << ", retcode=" << iret << endl;
    }
    else {
        cout << "Setting the AOI Left shift to" << left_shift << endl;
    }



    AT_64 img_orig_height,height_max;//already take into account of 2x2 binning
    AT_GetIntMax(_handle, L"AOIHeight", &height_max);
    cout << "Min AOI width is " << height_max << endl;
    AT_GetInt(_handle, L"AOIHeight", &img_orig_height);
    iret = AT_SetInt(_handle,L"AOIHeight",CCDSIZEY);
    AT_64 height_actual;
    AT_GetInt(_handle, L"AOIHeight", &height_actual);
    if (iret != AT_SUCCESS ) {
        cout << "Error setting the AOI Height to " << CCDSIZEY << ", retcode=" << iret << endl;
    }
    else {
        cout << "Setting the AOI Height to " << height_actual << endl;
    }





    AT_64 top_shift;
    //already take into account of the 2x2 binning 
    top_shift=img_orig_height- height_actual +1;
    iret = AT_SetInt(_handle,L"AOITop",top_shift); 
    if (iret != AT_SUCCESS ) {
        cout << "Error setting the AOI Top shift to " << top_shift << ", retcode=" << iret << endl;
    }
    else {
        cout << "Setting the AOI Top shift to" << top_shift << endl;
    }

    return iret;



}

void T2Cam_Close(CamData* MyCamera, AT_H _handle) {
    T2Cam_TurnOff(MyCamera, _handle);
    T2Cam_CloseLib();
}

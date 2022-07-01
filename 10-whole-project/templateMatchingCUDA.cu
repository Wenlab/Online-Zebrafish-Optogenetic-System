#include"templateMatchingCUDA.cuh"
#include"initANDcheck.h"



using namespace std;
using namespace chrono;
//ͼ�񼸺α任C++ʵ��--����ƽ�ƣ���ת�����У�����
//https://blog.csdn.net/duiwangxiaomi/article/details/109532590


//npp��ͼ����ת����nppiRotate_32f_C1Rʹ�÷�����
//����ת�Ƕ�Ϊ����ͼ������½������ȿ�ʼ���ģ�˳ʱ����ת�����Ͻǿ�ʼ���ģ���ʱ����ת
//����ת�Ƕ�Ϊ����ͼ������½������ȿ�ʼ���ģ���ʱ����ת�����Ͻǿ�ʼ���ģ�˳ʱ����ת
float *ObjRecon_imrotate3(float *ObjRecon, double nAngle)
{
	//float *input_image = new float[200*200];
	float *imageRotated3D = new float[200 * 200 * 50];
	//for (int i = 0; i < 200 * 200; i++)
	//{
	//	input_image[i] = ObjRecon[i];
	//}

	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 200;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);
	//check(cudaMemcpy(input_image_gpu, input_image, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 50; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	////��תǰ��ĵ�һ�����ηֱ�д��������
	//float *ObjRecon_1 = new float[200 * 200];
	//float *imageRotated3D_1 = new float[200 * 200];
	//for (int i = 0; i < 200*200; i++)
	//{
	//	ObjRecon_1[i] = ObjRecon[i];
	//	imageRotated3D_1[i] = imageRotated3D[i];
	//}
	//GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	//GDALDataset *ds1 = pDriver->Create("ObjRecon_1_c", 200, 200, 1, GDT_Float32, NULL);
	//GDALDataset *ds2 = pDriver->Create("imageRotated3D_1_c", 200, 200, 1, GDT_Float32, NULL);
	//if ((ds1 == NULL) || (ds2 == NULL))
	//{
	//	cout << "create ObjRecon_1 imageRotated3D_1 output_file error!" << endl;
	//	system("pause");
	//	return 0;
	//}
	////��ͼ������Ͻ�һ��һ�е�д
	//ds1->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, ObjRecon_1, 200, 200, GDT_Float32, 0, 0);
	//ds2->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, 200, 200, imageRotated3D_1, 200, 200, GDT_Float32, 0, 0);
	//GDALClose(ds1);
	//GDALClose(ds2);



	return imageRotated3D;
}

//����X����ת
float *ObjRecon_imrotate3_X(float *imageRotated3D, double nAngle)
{
	//imageRotated3D(200*200*50)ת���ɣ��б�ɲ��Σ����α���У��б�ɷ��ţ����200��*50��*200����
	float *ObjRecon = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//�������ѭ�����������ѭ��
	{
		for (int j = 0; j < 200; j++)//�����ѭ�����������ѭ����������
		{
			for (int k = 0; k < 50; k++)//�����ѭ��������Ĳ���ѭ��
			{
				//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
				ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
			}
		}
	}

	float *imageRotated3D_rotate = new float[200 * 50 * 200];

	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, ObjRecon + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyHostToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_rotate + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToHost), "output_image cudaMemcpy Error");
	}

	//�ٱ任��ԭ����ά�ȷֲ�
	//200��*50��*200���Ρ�>>200��*200��*50��
	float *imageRotated3D_rotate_return = new float[200 * 50 * 200];
	for (int i = 0; i < 200; i++)//�������ѭ�����������ѭ��
	{
		for (int j = 0; j < 200; j++)//�����ѭ�����������ѭ����������
		{
			for (int k = 0; k < 50; k++)//�����ѭ��������Ĳ���ѭ��
			{
				imageRotated3D_rotate_return[(49 - k) * 200 * 200 + (199 - j) * 200 + i] = imageRotated3D_rotate[i * 200 * 50 + j * 50 + k];
			}
		}
	}

	return imageRotated3D_rotate_return;
}

//�ز�����ָ������
int reSampleGDAL(const char* pszSrcFile, const char* pszOutFile, int newWidth, int newHeight, GDALResampleAlg eResample)
{
	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDataset* pDSrc = (GDALDataset*)GDALOpen(pszSrcFile, GA_Update);
	if (pDSrc == NULL)
	{
		cout << "���ļ�ʧ�ܣ�" << endl;
		return -1;
	}
	int width = pDSrc->GetRasterXSize();
	int height = pDSrc->GetRasterYSize();
	int nBandCount = pDSrc->GetRasterCount();
	GDALDataType dataType = pDSrc->GetRasterBand(1)->GetRasterDataType();
	char* pszSrcWKT = const_cast<char*>(pDSrc->GetProjectionRef());
	double dGeoTrans[6] = { 0 };
	pDSrc->GetGeoTransform(dGeoTrans);
	double dOldGeoTrans0 = dGeoTrans[0];
	//���û��ͶӰ����Ϊ����һ��    
	if (strlen(pszSrcWKT) <= 0)
	{
		OGRSpatialReference oSRS;
		oSRS.SetWellKnownGeogCS("WGS84");
		oSRS.exportToWkt(&pszSrcWKT);
		pDSrc->SetProjection(pszSrcWKT);
		dGeoTrans[0] = 30.0;
		dGeoTrans[3] = 30.0;
		dGeoTrans[1] = 0.00001;
		dGeoTrans[5] = -0.00001;
		pDSrc->SetGeoTransform(dGeoTrans);
	}


	/*************** �����ز�����ĳߴ���ȡͶӰ��Ϣ **************/
	float fResX = (1.0 * newWidth) / width;
	float fResY = (1.0 * newHeight) / height;
	//��ȡ�ز�����Ϣ
	dGeoTrans[1] = dGeoTrans[1] / fResX;
	dGeoTrans[5] = dGeoTrans[5] / fResY;
	int nNewWidth = static_cast<int>(newWidth + 0.5);
	int nNewHeight = static_cast<int>(newHeight + 0.5);
	//��������ļ�
	GDALDriver* pDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (pDriver == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		cout << "�����ļ�ʧ�ܣ�" << endl;
		return -2;
	}
	//����������ݼ�  
	GDALDataset* pDDst = pDriver->Create(pszOutFile, nNewWidth, nNewHeight, nBandCount, dataType, NULL);
	if (pDDst == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		cout << "�����ļ�ʧ�ܣ�" << endl;
		return -2;
	}
	pDDst->SetProjection(pszSrcWKT);
	pDDst->SetGeoTransform(dGeoTrans);



	//�����ز���ǰ��Ķ�Ӧ��ϵ��׼������...
	void* hTransformArg = GDALCreateGenImgProjTransformer2((GDALDatasetH)pDSrc, (GDALDatasetH)pDDst, NULL); //GDALCreateGenImgProjTransformer((GDALDatasetH) pDSrc,pszSrcWKT,(GDALDatasetH) pDDst,pszSrcWKT,FALSE,0.0,1);  
	if (hTransformArg == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);
		cout << "��������г���" << endl;
		return -3;
	}

	GDALWarpOptions* psWo = GDALCreateWarpOptions();
	psWo->papszWarpOptions = CSLDuplicate(NULL);
	psWo->eWorkingDataType = dataType;
	psWo->eResampleAlg = eResample;
	psWo->hSrcDS = (GDALDatasetH)pDSrc;
	psWo->hDstDS = (GDALDatasetH)pDDst;
	psWo->pfnTransformer = GDALGenImgProjTransform;
	psWo->pTransformerArg = hTransformArg;
	psWo->nBandCount = nBandCount;
	psWo->panSrcBands = (int*)CPLMalloc(nBandCount * sizeof(int));
	psWo->panDstBands = (int*)CPLMalloc(nBandCount * sizeof(int));
	for (int i = 0; i < nBandCount; i++)
	{
		psWo->panSrcBands[i] = i + 1;
		psWo->panDstBands[i] = i + 1;
	}


	GDALWarpOperation oWo;
	if (oWo.Initialize(psWo) != CE_None)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);

		cout << "��������г���" << endl;
		return -3;
	}

	oWo.ChunkAndWarpImage(0, 0, nNewWidth, nNewHeight);

	//�ƺ�
	GDALDestroyGenImgProjTransformer(hTransformArg);
	GDALDestroyWarpOptions(psWo);
	GDALFlushCache(pDDst);
	GDALClose((GDALDatasetH)pDSrc);
	GDALClose((GDALDatasetH)pDDst);
	return 0;
}

//�ز�����ָ������
int reSampleGDAL_1(float *ArrayBand, int width, int height, int nBandCount, GDALDataType dataType,
	float *ArrayBand_out, int newWidth, int newHeight, GDALResampleAlg eResample)
{
	//��������ļ�
	GDALDriver *pDriver = GetGDALDriverManager()->GetDriverByName("MEM");
	if (pDriver == NULL)
	{
		cout << "����GDALDriverʧ�ܣ�" << endl;
		return -2;
	}
	GDALDataset *pDSrc = pDriver->Create("", width, height, nBandCount, dataType, NULL);
	pDSrc->RasterIO(GF_Write, 0, 0, width, height, ArrayBand, width, height, dataType, nBandCount, NULL, 0, 0, 0);
	//����ͶӰ
	char* pszSrcWKT;
	double dGeoTrans[6] = { 0 };
	OGRSpatialReference oSRS;
	oSRS.SetWellKnownGeogCS("WGS84");
	oSRS.exportToWkt(&pszSrcWKT);
	pDSrc->SetProjection(pszSrcWKT);
	dGeoTrans[0] = 30.0;
	dGeoTrans[3] = 30.0;
	dGeoTrans[1] = 0.00001;
	dGeoTrans[5] = -0.00001;
	pDSrc->SetGeoTransform(dGeoTrans);

	/*************** �����ز�����ĳߴ���ȡͶӰ��Ϣ **************/
	float fResX = (1.0 * newWidth) / width;
	float fResY = (1.0 * newHeight) / height;
	//��ȡ�ز�����Ϣ
	dGeoTrans[1] = dGeoTrans[1] / fResX;
	dGeoTrans[5] = dGeoTrans[5] / fResY;
	int nNewWidth = static_cast<int>(newWidth + 0.5);
	int nNewHeight = static_cast<int>(newHeight + 0.5);

	//����������ݼ�  
	//GDALDriver* pDriver1 = GetGDALDriverManager()->GetDriverByName("GTiff");
	GDALDataset* pDDst = pDriver->Create("",
		nNewWidth, nNewHeight, nBandCount, dataType, NULL);
	if (pDDst == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		cout << "�����ļ�ʧ�ܣ�" << endl;
		return -2;
	}
	pDDst->SetProjection(pszSrcWKT);
	pDDst->SetGeoTransform(dGeoTrans);



	//�����ز���ǰ��Ķ�Ӧ��ϵ��׼������...
	void* hTransformArg = GDALCreateGenImgProjTransformer2((GDALDatasetH)pDSrc, (GDALDatasetH)pDDst, NULL); //GDALCreateGenImgProjTransformer((GDALDatasetH) pDSrc,pszSrcWKT,(GDALDatasetH) pDDst,pszSrcWKT,FALSE,0.0,1);  
	if (hTransformArg == NULL)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);
		cout << "��������г���" << endl;
		return -3;
	}

	GDALWarpOptions* psWo = GDALCreateWarpOptions();
	psWo->papszWarpOptions = CSLDuplicate(NULL);
	psWo->eWorkingDataType = dataType;
	psWo->eResampleAlg = eResample;
	psWo->hSrcDS = (GDALDatasetH)pDSrc;
	psWo->hDstDS = (GDALDatasetH)pDDst;
	psWo->pfnTransformer = GDALGenImgProjTransform;
	psWo->pTransformerArg = hTransformArg;
	psWo->nBandCount = nBandCount;
	psWo->panSrcBands = (int*)CPLMalloc(nBandCount * sizeof(int));
	psWo->panDstBands = (int*)CPLMalloc(nBandCount * sizeof(int));
	for (int i = 0; i < nBandCount; i++)
	{
		psWo->panSrcBands[i] = i + 1;
		psWo->panDstBands[i] = i + 1;
	}


	GDALWarpOperation oWo;
	if (oWo.Initialize(psWo) != CE_None)
	{
		GDALClose((GDALDatasetH)pDSrc);
		GDALClose((GDALDatasetH)pDDst);

		cout << "��������г���" << endl;
		return -3;
	}

	oWo.ChunkAndWarpImage(0, 0, nNewWidth, nNewHeight);

	pDDst->RasterIO(GF_Read, 0, 0, nNewWidth, nNewHeight, ArrayBand_out, nNewWidth, nNewHeight, dataType, nBandCount, NULL, 0, 0, 0);


	//�ƺ�
	GDALDestroyGenImgProjTransformer(hTransformArg);
	GDALDestroyWarpOptions(psWo);
	GDALFlushCache(pDDst);
	GDALClose((GDALDatasetH)pDSrc);
	GDALClose((GDALDatasetH)pDDst);
	return 0;
}


__global__ void kernel_1(float *ObjRecon_gpu, int height, int width, float *image2D_XY_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//��ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//��ѭ��

	if (i < 200 && j < 200)
	{
		image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[i * 200 + j];
		for (int b = 0; b < 50; b++)//����ѭ��
		{
			if (image2D_XY_gpu[i * 200 + j] < ObjRecon_gpu[b * 200 * 200 + i * 200 + j])
			{
				image2D_XY_gpu[i * 200 + j] = ObjRecon_gpu[b * 200 * 200 + i * 200 + j];
			}
		}//����ѭ��
	}
}
__global__ void kernel_2(float *image2D_XY_gpu, int total, double image2D_XY_mean, float *img2DBW_XY_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if (image2D_XY_gpu[i] > image2D_XY_mean)
			img2DBW_XY_gpu[i] = 1.0;
			//img2DBW_XY_gpu[i] = 255;
		else
			img2DBW_XY_gpu[i] = 0.0;
	}

}
__global__ void kernel_3(float *template_roXY_gpu, float *img2DBW_XY_gpu, int rotationAngleXY_size, double *err_XY_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < rotationAngleXY_size)
	{
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 200; k++)//��ѭ��
			{
				sum_temp += (template_roXY_gpu[i * 200 * 200 + j * 200 + k] - img2DBW_XY_gpu[j * 200 + k])*
					(template_roXY_gpu[i * 200 * 200 + j * 200 + k] - img2DBW_XY_gpu[j * 200 + k]);
			}
		}
		err_XY_gpu[i] = sum_temp / (200 * 200);
	}
}

__global__ void kernel_4(float *imageRotated3D_gpu, float *image2D_YZ_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//����ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//��ѭ��

	if (i < 50 && j < 200)
	{
		image2D_YZ_gpu[i * 200 + j] = -FLT_MAX;
		for (int k = 0; k < 200; k++)//��ѭ������һ�е����ֵ
		{
			if (image2D_YZ_gpu[i * 200 + j] < imageRotated3D_gpu[i * 200 * 200 + j * 200 + k])
			{
				image2D_YZ_gpu[i * 200 + j] = imageRotated3D_gpu[i * 200 * 200 + j * 200 + k];
			}
		}
	}
}
__global__ void kernel_5(float *image2D_YZ_gpu, double image2D_YZ_mean, float *img2DBW_YZ_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < 200 * 50)
	{
		if (image2D_YZ_gpu[i] > image2D_YZ_mean)
			img2DBW_YZ_gpu[i] = 1.0;
		else
			img2DBW_YZ_gpu[i] = 0.0;
	}
}
__global__ void kernel_6(float *template_roYZ_gpu, float *img2DBW_YZ_gpu, int rotationAngleYZ_size, double *err_YZ_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < rotationAngleYZ_size)
	{
		//������������ľ������
		double sum_temp = 0;
		for (int j = 0; j < 200; j++)//��ѭ��
		{
			for (int k = 0; k < 50; k++)//��ѭ��
			{
				//template_roYZ��200��*50��*31���Σ����������У�img2DBW_YZ�����������е�
				sum_temp += (template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j])*
					(template_roYZ_gpu[i * 200 * 50 + j * 50 + k] - img2DBW_YZ_gpu[k * 200 + j]);
			}
		}
		err_YZ_gpu[i] = sum_temp / (200 * 50);
	}
}
//ά�ȱ任
__global__ void kernel_7(float *imageRotated3D_gpu, float *imageRotated3D_gpu_1)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//�������ѭ�����������ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//�����ѭ�����������ѭ����������
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//�����ѭ��������Ĳ���ѭ��

	if (i < 200 && j < 200 && k < 50)
	{
		//ObjRecon[i * 200 * 50 + j * 50 + k] = imageRotated3D[199-j][i][49-k];
		imageRotated3D_gpu_1[i * 200 * 50 + j * 50 + k] = imageRotated3D_gpu[(49 - k) * 200 * 200 + (199 - j) * 200 + i];
	}
}
//����X����ת
void ObjRecon_imrotate3_X_gpu(float *imageRotated3D_gpu_1, double nAngle, float *imageRotated3D_gpu_2)
{
	NppiSize Input_Size;//����ͼ���������
	Input_Size.width = 200;
	Input_Size.height = 50;
	/* �����Դ棬��ԭͼ�����Դ� */
	int nSrcPitchCUDA = Input_Size.width * sizeof(float);//ÿ����ռ���ֽ���
	float *input_image_gpu;
	check1(cudaMalloc((void**)&input_image_gpu, sizeof(float)*Input_Size.width*Input_Size.height), "input_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	/* ������ת�󳤿� */
	NppiRect Input_ROI;//�ض��������ת���൱�ڲü�ͼ���һ�飬���β���ȫ��ͼ��
	Input_ROI.x = Input_ROI.y = 0;
	Input_ROI.width = Input_Size.width;
	Input_ROI.height = Input_Size.height;
	double aBoundingBox[2][2];
	nppiGetRotateBound(Input_ROI, aBoundingBox, nAngle, 0, 0);
	int bb = ((int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0])) - Input_ROI.width) / 2 + aBoundingBox[0][0];//��ʼ��
	int cc = ((int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1])) - Input_ROI.height) / 2 + aBoundingBox[0][1];//��ʼ��
	aBoundingBox[0][0] = bb;//��ʼ��
	aBoundingBox[0][1] = cc;//��ʼ��
	NppiSize Output_Size;
	Output_Size.width = (int)ceil(fabs(aBoundingBox[1][0] - aBoundingBox[0][0]));
	Output_Size.height = (int)ceil(fabs(aBoundingBox[1][1] - aBoundingBox[0][1]));
	Output_Size.width = Input_Size.width;
	Output_Size.height = Input_Size.height;


	/* ת�����ͼ���Դ���� */
	int nDstPitchCUDA = Output_Size.width * sizeof(float);
	float *output_image_gpu;
	check1(cudaMalloc((void**)&output_image_gpu, sizeof(float)*Output_Size.width*Output_Size.height), "output_image_gpu cudaMalloc Error", __FILE__, __LINE__);


	//�������Ȥ���Ĵ�С���൱�ڰ����ͼ���ٲü�һ�飬Ӧ������������û���ԣ��������
	NppiRect Output_ROI;
	Output_ROI.x = 0; Output_ROI.y = 0;
	Output_ROI.width = Input_Size.width;
	Output_ROI.height = Input_Size.height;

	for (int i = 0; i < 200; i++)
	{
		check(cudaMemcpy(input_image_gpu, imageRotated3D_gpu_1 + Input_Size.width*Input_Size.height * i, sizeof(float)*Input_Size.width*Input_Size.height, cudaMemcpyDeviceToDevice), "input_image_gpu cudaMemcpy Error");
		/* ������ת */
		NppStatus nppRet = nppiRotate_32f_C1R(input_image_gpu, Input_Size, nSrcPitchCUDA, Input_ROI,
			output_image_gpu, nDstPitchCUDA, Output_ROI, nAngle, -aBoundingBox[0][0], -aBoundingBox[0][1], NPPI_INTER_NN);
		assert(nppRet == NPP_NO_ERROR);
		check(cudaMemcpy(imageRotated3D_gpu_2 + Input_Size.width*Input_Size.height * i, output_image_gpu, sizeof(float) * Output_Size.width*Output_Size.height, cudaMemcpyDeviceToDevice), "output_image cudaMemcpy Error");
	}
}
//�ٱ任��ԭ����ά��
__global__ void kernel_8(float *imageRotated3D_gpu_2, float *imageRotated3D_gpu)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;//�������ѭ�����������ѭ��
	const int j = blockDim.y * blockIdx.y + threadIdx.y;//�����ѭ�����������ѭ����������
	const int k = blockDim.z * blockIdx.z + threadIdx.z;//�����ѭ��������Ĳ���ѭ��

	if (i < 200 && j < 200 && k < 50)//�������ѭ�����������ѭ��
	{
		imageRotated3D_gpu[(49 - k) * 200 * 200 + (199 - j) * 200 + i] = imageRotated3D_gpu_2[i * 200 * 50 + j * 50 + k];
	}
}
__global__ void kernel_9(float *imageRotated3D_gpu, double imageRotated3D_x_mean, int *BWObjRecon_gpu)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < 200 * 200 * 50)
	{
		if (imageRotated3D_gpu[i] > imageRotated3D_x_mean)
			BWObjRecon_gpu[i] = 1;
		else
			BWObjRecon_gpu[i] = 0;
	}
}
__global__ void kernel_10(float *imageRotated3D_gpu, float *ObjReconRed_gpu, int XObj, int YObj, int ZObj, int CentroID0, int CentroID2)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;//XObj
	const int y = blockDim.y * blockIdx.y + threadIdx.y;//YObj
	const int z = blockDim.z * blockIdx.z + threadIdx.z;//ZObj

	if (z < ZObj && x < XObj && y < YObj)
	{
		ObjReconRed_gpu[z*XObj*YObj + y * XObj + x] = imageRotated3D_gpu[z * 200 * 200 + (CentroID0 - 61 + y) * 200 + CentroID2 - 38 + x];
	}

}
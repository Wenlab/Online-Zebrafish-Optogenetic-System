#include "header.cuh"
#include <chrono>//标准模板库中与时间有关的头文件
using namespace chrono;
//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("GPU Parament：\n");
	printf(" Device Name : %s.\n", prop.name);
	printf(" totalGlobalMem : %I64d.\n", prop.totalGlobalMem);
	printf(" sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf(" regsPerBlock : %d.\n", prop.regsPerBlock);
	printf(" warpSize : %d.\n", prop.warpSize);
	printf(" memPitch : %d.\n", prop.memPitch);
	printf(" maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf(" maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf(" maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf(" totalConstMem : %d.\n", prop.totalConstMem);
	printf(" major.minor : %d.%d.\n", prop.major, prop.minor);
	printf(" clockRate : %d.\n", prop.clockRate);
	printf(" textureAlignment : %d.\n", prop.textureAlignment);
	printf(" deviceOverlap : %d.\n", prop.deviceOverlap);
	printf(" multiProcessorCount : %d.\n", prop.multiProcessorCount);
	std::printf(" CUDA core: %d\r\n", 2 * prop.multiProcessorCount* prop.maxThreadsPerMultiProcessor / prop.warpSize);
	printf("\n\n");
}
//CUDA 初始化
bool InitCUDA()
{
	int count;
	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//打印设备信息
		printDeviceProp(prop);
		//获得显卡的时钟频率
		clockRate = prop.clockRate;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}
void check(cudaError_t res, string warningstring)
{
	if (res != cudaSuccess)
	{
		printf((warningstring + " !\n").c_str());
		system("pause");
		exit(0);
	}
}
//查看GPU运行是否正确
void checkGPUStatus(cudaError_t cudaGetLastError, string warningstring)
{
	if (cudaGetLastError != cudaSuccess)
	{
		printf("\n\n");
		printf((warningstring + " !\n").c_str());
		fprintf(stderr, "%s\n", cudaGetErrorString(cudaGetLastError));
		system("pause");
		exit(0);
	}
}

__global__ void Zhuan_Complex_kernel(float *PSF_1_gpu, cufftComplex *PSF_1_gpu_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		PSF_1_gpu_Complex[i].x = PSF_1_gpu[i];
		PSF_1_gpu_Complex[i].y = 0;
	}
}
__global__ void PSF_unshort(float *PSF_1_gpu, unsigned short *PSF, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if (PSF_1_gpu[i] < 0)
		{
			PSF[i] = 0;
		}
		else if (PSF_1_gpu[i] > 65535)
		{
			PSF[i] = 65535;
		}
		else
		{
			PSF[i] = (int)(PSF_1_gpu[i] + 0.5);
		}
	}
}
__global__ void initial_kernel_1(float *ImgEst, float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		ImgEst[i] = 0;
		Ratio[i] = 1;
	}
}
__global__ void gpuObjRecon_fuzhi(float *gpuObjRecon, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecon[i] = 1;
	}
}
__global__ void initial_kernel_3(float *gpuObjRecROI, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecROI[i] = 1;
	}
}
__global__ void ImgExp_ge(unsigned short *Img_gpu, int BkgMean, float *ImgExp, int total)
{
	//把差的结果小于0的变成0，大于0的四舍五入，大于65535的变成65535
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		if ((Img_gpu[i] - BkgMean) < 0)
		{
			ImgExp[i] = 0;
		}
		else if ((Img_gpu[i] - BkgMean) > 65535)
		{
			ImgExp[i] = 65535;
		}
		else
		{
			ImgExp[i] = (int)((Img_gpu[i] - BkgMean) + 0.5);
		}
	}
}
__global__ void Ratio_fuzhi(float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio[i] = 1;
	}
}
__global__ void OTF_mul_gpuObjRecon_Complex(cufftComplex *OTF, cufftComplex *gpuObjRecon_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		float aaa = OTF[i].x*gpuObjRecon_Complex[i].x - OTF[i].y*gpuObjRecon_Complex[i].y;//相乘的实部结果
		float bbb = OTF[i].x*gpuObjRecon_Complex[i].y + OTF[i].y*gpuObjRecon_Complex[i].x;//相乘的虚部结果
		gpuObjRecon_Complex[i].x = aaa;
		gpuObjRecon_Complex[i].y = bbb;
	}
}
__global__ void ifftshift_real_max(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	int lie_half = PSF_size_2 / 2;
	if (i < PSF_size_1 / 2 && j < PSF_size_2 && k < PSF_size_3)
	{
		//实现图像的ifftshift+real+max，即：将图像划分成4个象限，第一和第三平移交换，第二和第四平移交换，
		float_temp[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * 512] = OTF[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j].x >= 0 ? OTF[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j].x : 0;
		float_temp[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j] = OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2].x >= 0 ? OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2].x : 0;
	}
}
__global__ void ifftshift(cufftComplex *OTF, float *float_temp, int PSF_size_1, int PSF_size_2, int PSF_size_3, cufftComplex *OTF_ifftshift)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	int lie_half = PSF_size_2 / 2;
	if (i < PSF_size_1 / 2 && j < PSF_size_2 && k < PSF_size_3)
	{
		//实现图像的ifftshift，即：将图像划分成4个象限，第一和第三平移交换，第二和第四平移交换
		OTF_ifftshift[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * 512] = OTF[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j];
		OTF_ifftshift[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j] = OTF[k*PSF_size_1*PSF_size_2 + (i + PSF_size_1 / 2)*PSF_size_2 + j + lie_half - j / lie_half * PSF_size_2];
	}
}
__global__ void float_temp_sum(float *float_temp, float *ImgEst, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < PSF_size_1 && j < PSF_size_2)
	{
		ImgEst[i*PSF_size_2 + j] = 0;
		for (int k = 0; k < PSF_size_3; k++)
		{
			ImgEst[i*PSF_size_2 + j] += float_temp[k*PSF_size_1*PSF_size_2 + (i*PSF_size_2 + j)];
		}
	}
}
__global__ void Ratio_fuzhi_2(float *ImgExp, float *ImgEst, float Tmp, int SNR, float *Ratio, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio[i] = ImgExp[i]/(ImgEst[i] + Tmp / SNR);
	}
}
__global__ void Ratio_Complex_ge(float *ImgExp, float *ImgEst, float Tmp, int SNR, cufftComplex *Ratio_Complex, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		Ratio_Complex[i].x = ImgExp[i] / (ImgEst[i] + Tmp / SNR);
		Ratio_Complex[i].y = 0;
	}
}
__global__ void fftRatio_ge(cufftComplex *Ratio_Complex, cufftComplex *fftRatio, int PSF_size_1, int PSF_size_2, int PSF_size_3)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	const int k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < PSF_size_1 && j < PSF_size_2 && k < PSF_size_3)
	{
		fftRatio[k*PSF_size_1*PSF_size_2 + i*PSF_size_2 + j] = Ratio_Complex[i*PSF_size_2 + j];
	}
}
__global__ void fftceshi_gpu_fuzhi(cufftComplex *PSF_1_gpu_Complex, cufftComplex *fftceshi_gpu, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		fftceshi_gpu[i] = PSF_1_gpu_Complex[i];
	}
}
__global__ void ifft2_divide(cufftComplex *OTF, int total, int scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		OTF[i].x = OTF[i].x / scale;
		OTF[i].y = OTF[i].y / scale;
	}
}
__global__ void real_multiply(float *gpuObjRecon, float *float_temp, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		gpuObjRecon[i] = gpuObjRecon[i] * float_temp[i];
	}
}
__global__ void fftRatio_mul_conjOTF(cufftComplex *fftRatio, cufftComplex *OTF, int total)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < total)
	{
		float aaa = fftRatio[i].x*OTF[i].x + fftRatio[i].y*OTF[i].y;//共轭相乘的实部结果
		float bbb = -fftRatio[i].x*OTF[i].y + fftRatio[i].y*OTF[i].x;//共轭相乘的虚部结果
		fftRatio[i].x = aaa;
		fftRatio[i].y = bbb;
	}
}


int main()
{
	//int geshu = 32;
	//cufftComplex *aa = new cufftComplex[geshu];
	//for (int i = 0; i < geshu; i++)
	//{
	//	aa[i].x = i + 1; aa[i].x += aa[i].x / 10;
	//	aa[i].y = 0;
	//}
	//cufftComplex *aa_gpu;
	//check(cudaMalloc((void**)&aa_gpu, geshu * sizeof(cufftComplex)), "aa_gpu cudaMalloc Error");
	//check(cudaMemcpy(aa_gpu, aa, geshu * sizeof(cufftComplex), cudaMemcpyHostToDevice), "aa_gpu cudaMemcpy Error");
	//cufftComplex *aa_gpu_fft2;
	//check(cudaMalloc((void**)&aa_gpu_fft2, geshu * sizeof(cufftComplex)), "aa_gpu_fft2 cudaMalloc Error");
	//cufftHandle plan_aa;
	//cufftResult res1 = cufftPlan2d(&plan_aa, 4, 8, CUFFT_C2C);
	//res1 = cufftExecC2C(plan_aa, aa_gpu, aa_gpu_fft2, CUFFT_FORWARD);
	//cudaDeviceSynchronize();
	//checkGPUStatus(cudaGetLastError(), "aa_gpu cufftExecC2C Error");
	//check(cudaMemcpy(aa, aa_gpu_fft2, geshu * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "aa_gpu cudaMemcpy Error");
	//float qq = 0; float ww = 0;
	//for (int i = 0; i < geshu; i++)
	//{
	//	qq += aa[i].x; ww += aa[i].y;
	//}

	//开始计时
	auto time_start = system_clock::now();

	const char *PSF_1_file = "F:/matlab-cuda-20220125/PSF_1_zhuanzhi_float.dat";//matlab中保存出来的float类型
	const char *X31_file = "F:/matlab-cuda-20220125/r20210924_2_X31_resize.tif";
	const char *OutFile = "F:/matlab-cuda-20220125/result_C.dat";
	FILE *PSF_1_fid = fopen(PSF_1_file, "rb");
	if (PSF_1_fid == NULL)
	{
		cout << "PSF_1_file open failed!" << endl;
		system("pause");
		return 0;
	}
	float *PSF_1 = new float[PSF_size_1*PSF_size_2*PSF_size_3]();
	fread(PSF_1, sizeof(float), PSF_size_1*PSF_size_2*PSF_size_3, PSF_1_fid);
	//float ccc_sum = 0;
	//for (int i = 0; i < PSF_size_3; i++)
	//{
	//	ccc_sum = 0.0;
	//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
	//	{
	//		ccc_sum += PSF_1[j];
	//	}
	//	cout << "第" << i+1 << "个波段的和是：" << fixed << ccc_sum << endl;
	//}

	//使用GDAL读取tif，使用的是matlab重采样好的数据
	GDALAllRegister(); OGRRegisterAll();
	//设置支持中文路径
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	GDALDataset* poSrcDS = (GDALDataset*)GDALOpen(X31_file, GA_ReadOnly);    //打开栅格图像
	if (poSrcDS == NULL)
	{
		cout << "X31_file open failed!" << endl;
		return;
	}
	int wheight = poSrcDS->GetRasterYSize();//高
	int wwidth = poSrcDS->GetRasterXSize();//宽
	int bandNum = poSrcDS->GetRasterCount();//波段数
	GDALDataType dataType = poSrcDS->GetRasterBand(1)->GetRasterDataType();//类型
	unsigned short *Img = new unsigned short[PSF_size_1*PSF_size_2]();
	for (int i = 0; i < bandNum; i++)
	{
		//////////////////////////////////////读取，始列，始行，列数，行数，指针，列数，行数，数据类型
		poSrcDS->GetRasterBand(i + 1)->RasterIO(GF_Read, 0, 0, wwidth, wheight, Img, PSF_size_1, PSF_size_2, dataType, 0, 0);
	}
	GDALClose(poSrcDS);

	/*-------准备工作：申请内存、显存，fft句柄--------------------------*/
	const int rank = 2;//维数
	int n[rank] = { PSF_size_1, PSF_size_2 };//n*m
	int *inembed = n;//输入的数组size
	int istride = 1;//数组内数据连续，为1
	int idist = n[0] * n[1];//1个数组的内存大小
	int *onembed = n;//输出是一个数组的size
	int ostride = 1;//每点DFT后数据连续则为1
	int odist = n[0] * n[1];//输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
	int batch = PSF_size_3;//批量处理的批数
	cufftHandle fftplanfwd;//创建句柄
	cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

	float *PSF_1_gpu;
	check(cudaMalloc((void**)&PSF_1_gpu, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "PSF_1_gpu cudaMalloc Error");
	cufftComplex *PSF_1_gpu_Complex;
	check(cudaMalloc((void**)&PSF_1_gpu_Complex, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "PSF_1_gpu_Complex cudaMalloc Error");
	cufftComplex *OTF;
	check(cudaMalloc((void**)&OTF, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "OTF cudaMalloc Error");
	float *ImgEst;
	check(cudaMalloc((void**)&ImgEst, PSF_size_1*PSF_size_2 * sizeof(float)), "ImgEst cudaMalloc Error");
	float *Ratio;
	check(cudaMalloc((void**)&Ratio, PSF_size_1*PSF_size_2 * sizeof(float)), "Ratio cudaMalloc Error");
	float *gpuObjRecon;
	check(cudaMalloc((void**)&gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "Ratio cudaMalloc Error");
	float *gpuObjRecROI;
	check(cudaMalloc((void**)&gpuObjRecROI, ROISize * 2 * ROISize * 2 * PSF_size_3 * sizeof(float)), "gpuObjRecROI cudaMalloc Error");
	unsigned short *Img_gpu;
	check(cudaMalloc((void**)&Img_gpu, PSF_size_1*PSF_size_2 * sizeof(unsigned short)), "Img_gpu cudaMalloc Error");
	float *ImgExp;
	check(cudaMalloc((void**)&ImgExp, PSF_size_1*PSF_size_2 * sizeof(float)), "ImgExp cudaMalloc Error");
	cufftComplex *gpuObjRecon_Complex;
	check(cudaMalloc((void**)&gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "gpuObjRecon_Complex cudaMalloc Error");
	float *float_temp;
	check(cudaMalloc((void**)&float_temp, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(float)), "float_temp cudaMalloc Error");
	cufftComplex *Ratio_Complex;
	check(cudaMalloc((void**)&Ratio_Complex, PSF_size_1*PSF_size_2 * sizeof(cufftComplex)), "Ratio_Complex cudaMalloc Error");
	cufftComplex *fftRatio;
	check(cudaMalloc((void**)&fftRatio, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)), "fftRatio cudaMalloc Error");




	/*---开始处理数据------------------------------------------------------------------------*/
	auto time_1 = system_clock::now();
	/*-----对PSF_1做fft2-------*/
	check(cudaMemcpy(PSF_1_gpu, PSF_1, PSF_size_1*PSF_size_2*PSF_size_3*sizeof(float), cudaMemcpyHostToDevice), "PSF_1_gpu cudaMemcpy Error");
	//转换成复数，虚部是0
	Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (PSF_1_gpu, PSF_1_gpu_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu Zhuan_Complex_kernel Error");
	///////----测试一个波段的fft2是否正确-----------
	//cufftComplex *fftceshi_gpu;
	//check(cudaMalloc((void**)&fftceshi_gpu, PSF_size_1*PSF_size_2*sizeof(cufftComplex)), "fftceshi_gpu cudaMalloc Error");
	//cufftComplex *fftceshi_gpu_shuchu;
	//check(cudaMalloc((void**)&fftceshi_gpu_shuchu, PSF_size_1*PSF_size_2 * sizeof(cufftComplex)), "fftceshi_gpu_shuchu cudaMalloc Error");
	////转成复数，虚部是0
	//fftceshi_gpu_fuzhi << <(PSF_size_1*PSF_size_2 - 1) / 256 + 1, 256 >> > (PSF_1_gpu_Complex, fftceshi_gpu, PSF_size_1*PSF_size_2);
	//cudaDeviceSynchronize();
	//checkGPUStatus(cudaGetLastError(), "fftceshi_gpu Error");
	//cufftHandle plan;
	//cufftResult res = cufftPlan2d(&plan, PSF_size_1, PSF_size_2, CUFFT_C2C);
	//res = cufftExecC2C(plan, fftceshi_gpu, fftceshi_gpu_shuchu, CUFFT_FORWARD);
	//if (res != CUFFT_SUCCESS)
	//{
	//	cout << "fftceshi_gpu cufftExecC2C error:" << res << endl;
	//	system("pause");
	//	return;
	//}
	//
	////*----拷贝到内存查看是否正确-------*/
	//cufftComplex *abc = new cufftComplex[PSF_size_1*PSF_size_2];
	//check(cudaMemcpy(abc, fftceshi_gpu_shuchu, PSF_size_1*PSF_size_2*sizeof(cufftComplex), cudaMemcpyDeviceToHost), "abc cudaMemcpy Error");
	//float *abc_real = new float[PSF_size_1*PSF_size_2];
	//float *abc_imag = new float[PSF_size_1*PSF_size_2];
	//float abc_real_sum = 0; float abc_imag_sum = 0;
	//for (int j = 0; j < PSF_size_1*PSF_size_2; j++)
	//{
	//	abc_real[j] = abc[j].x; abc_real_sum += abs(abc_real[j]);
	//	abc_imag[j] = abc[j].y; abc_imag_sum += abs(abc_imag[j]);
	//}
	////求每一行的实部、虚部和
	//float *abc_real_lineSUM = new float[PSF_size_1]();
	//float *abc_imag_lineSUM = new float[PSF_size_1]();
	//for (int i = 0; i < PSF_size_1; i++)
	//{
	//	for (int j = 0; j < PSF_size_2; j++)
	//	{
	//		abc_real_lineSUM[i] += abs(abc_real[i*PSF_size_2 + j]);
	//		abc_imag_lineSUM[i] += abs(abc_imag[i*PSF_size_2 + j]);
	//	}
	//}

	////*----使用cufftPlanMany的方法进行批量二维fft---------------------*/
	cufftExecC2C(fftplanfwd, PSF_1_gpu_Complex, OTF, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu_Complex cufftExecC2C Error");
	//////结果拷贝到内存，查看是否和matlab一致
	//cufftComplex *OTF_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex)];
	//check(cudaMemcpy(OTF_cpu, OTF, PSF_size_1*PSF_size_2*PSF_size_3 * sizeof(cufftComplex), cudaMemcpyDeviceToHost), "OTF_cpu cudaMemcpy Error");
	//float aaa_sum = 0; float bbb_sum = 0;
	//for (int i = 0; i < PSF_size_3; i++)
	//{
	//	aaa_sum = 0; bbb_sum = 0;
	//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
	//	{
	//		aaa_sum += abs(OTF_cpu[j].x);
	//		bbb_sum += abs(OTF_cpu[j].y);
	//	}
	//	cout << "第" << i + 1 << "个波段的实部绝对值和：" << fixed << aaa_sum << "虚部绝对值和：" << fixed << bbb_sum <<endl;
	//}

	/////PSF_1转成uint16类型，存到数组PSF_gpu中
	//unsigned short *PSF_gpu;
	//check(cudaMalloc((void**)&PSF_gpu, sizeof(unsigned short) * PSF_size_1*PSF_size_2*PSF_size_3), "PSF_gpu cudaMalloc Error");
	//PSF_unshort << <blockNum_123, threadNum_123 >> > (PSF_1_gpu, PSF_gpu, PSF_size_1*PSF_size_2*PSF_size_3);
	//cudaDeviceSynchronize();
	//checkGPUStatus(cudaGetLastError(), "PSF_unshort Error");
	//////----拷贝到内存，查看是否正确
	//unsigned short *aaa = new unsigned short[PSF_size_1*PSF_size_2*PSF_size_3];
	//check(cudaMemcpy(aaa, PSF_gpu, sizeof(unsigned short) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "aaa cudaMemcpy Error");
	//float PSF_gpu_sum = 0;
	//for (int i = 0; i < PSF_size_1*PSF_size_2*PSF_size_3; i++)
	//{
	//	PSF_gpu_sum += aaa[i];
	//}

	////ImgEst赋值为0，Ratio赋值为1
	initial_kernel_1 << <blockNum_12, threadNum_12 >> > (ImgEst, Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_1 Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");
	////gpuObjRecROI赋值为1
	initial_kernel_3 << <blockNum_ROI, threadNum_ROI >> > (gpuObjRecROI, ROISize * 2 * ROISize * 2 * Nz);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_3 Error");

	//tif数据拷贝到显存
	check(cudaMemcpy(Img_gpu, Img, PSF_size_1*PSF_size_2 * sizeof(unsigned short), cudaMemcpyHostToDevice), "Img_gpu cudaMemcpy Error");
	//减去背景均值，结果放在float类型的数组ImgExp中
	ImgExp_ge << <blockNum_12, threadNum_12 >> > (Img_gpu, BkgMean, ImgExp, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "ImgExp_ge Error");

	//Ratio和gpuObjRecon的元素都赋值1
	Ratio_fuzhi << <blockNum_12, threadNum_12 >> > (Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "Ratio_fuzhi Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");


	//开始循环迭代
	for (int i = 0; i < ItN; i++)
	{
		////1、fft2(gpuObjRecon)
		Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (gpuObjRecon, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Zhuan_Complex_kernel Error");
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C Error");
		//cufftComplex *gpuObjRecon_Complex_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(gpuObjRecon_Complex_cpu, gpuObjRecon_Complex, sizeof(cufftComplex) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "gpuObjRecon_Complex_cpu cudaMemcpy Error");
		//float aaa_sum = 0; float bbb_sum = 0;
		//float gpuObjRecon_Complex_cpu_real_sum = 0; float gpuObjRecon_Complex_cpu_imag_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(gpuObjRecon_Complex_cpu[j].x);
		//		bbb_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//		gpuObjRecon_Complex_cpu_real_sum += abs(gpuObjRecon_Complex_cpu[j].x);
		//		gpuObjRecon_Complex_cpu_imag_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部和：" << fixed << aaa_sum << "虚部和：" << fixed << bbb_sum << endl;
		//}
		//cout << "gpuObjRecon_Complex_cpu实部和：" << fixed << gpuObjRecon_Complex_cpu_real_sum << "虚部和：" << fixed << gpuObjRecon_Complex_cpu_imag_sum << endl;

		////2、OTF.*fft2(gpuObjRecon_Complex)，结果放在gpuObjRecon_Complex中
		OTF_mul_gpuObjRecon_Complex << <blockNum_123, threadNum_123 >> > (OTF, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "2、OTF.*fft2(gpuObjRecon_Complex) Error");
		//cufftComplex *gpuObjRecon_Complex_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(gpuObjRecon_Complex_cpu, gpuObjRecon_Complex, sizeof(cufftComplex) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "gpuObjRecon_Complex_cpu cudaMemcpy Error");
		//float aaa_sum = 0; float bbb_sum = 0;
		//float gpuObjRecon_Complex_cpu_real_sum = 0; float gpuObjRecon_Complex_cpu_imag_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(gpuObjRecon_Complex_cpu[j].x);
		//		bbb_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//		gpuObjRecon_Complex_cpu_real_sum += abs(gpuObjRecon_Complex_cpu[j].x);
		//		gpuObjRecon_Complex_cpu_imag_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部和：" << fixed << aaa_sum << "虚部和：" << fixed << bbb_sum << endl;
		//}
		//cout << "gpuObjRecon_Complex_cpu实部和：" << fixed << gpuObjRecon_Complex_cpu_real_sum << "虚部和：" << fixed << gpuObjRecon_Complex_cpu_imag_sum << endl;

		////3、ifft2(OTF.*fft2(gpuObjRecon))，逆变换需要除以总像素个数
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C cufft_inverse Error");
		////4、除以像素总数才正确
		ifft2_divide << <blockNum_123, threadNum_123 >> > (gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex ifft2_divide Error");
		//cufftComplex *gpuObjRecon_Complex_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(gpuObjRecon_Complex_cpu, gpuObjRecon_Complex, sizeof(cufftComplex) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "gpuObjRecon_Complex_cpu cudaMemcpy Error");
		//float aaa_sum = 0; float bbb_sum = 0;
		//float gpuObjRecon_Complex_cpu_real_sum = 0; float gpuObjRecon_Complex_cpu_imag_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(gpuObjRecon_Complex_cpu[j].x); bbb_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//		gpuObjRecon_Complex_cpu_real_sum += abs(gpuObjRecon_Complex_cpu[j].x);
		//		gpuObjRecon_Complex_cpu_imag_sum += abs(gpuObjRecon_Complex_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部和：" << fixed << aaa_sum << "虚部和：" << fixed << bbb_sum << endl;
		//}
		//cout << "gpuObjRecon_Complex_cpu实部和：" << fixed << gpuObjRecon_Complex_cpu_real_sum << "虚部和：" << fixed << gpuObjRecon_Complex_cpu_imag_sum << endl;

		/*----第二遍gpuObjRecon_Complex的实部正确，虚部不正确，后面的代码只用了gpuObjRecon_Complex的实部，没用到虚部----------*/

		////5、ifftshift + real + max(,0)，获得实部矩阵float_temp，小于0的赋值0
		ifftshift_real_max << <grid, block >> > (gpuObjRecon_Complex, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "5、gpuObjRecon_Complex ifftshift_real_max Error");
		//float *float_temp_cpu = new float[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(float_temp_cpu, float_temp, sizeof(float) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "float_temp_cpu cudaMemcpy Error");
		//float aaa_sum = 0; float float_temp_cpu_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; 
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(float_temp_cpu[j]);
		//		float_temp_cpu_sum += abs(float_temp_cpu[j]);
		//	}
		//	cout << "第" << i + 1 << "个波段绝对值和：" << fixed << aaa_sum <<endl;
		//}
		//cout << "float_temp_cpu绝对值和：" << fixed << float_temp_cpu_sum << endl;

		////6、sum( ,3)，在第三维上计算和，返回PSF_size_1行PSF_size_2列的矩阵ImgEst
		float_temp_sum << <grid_sum, block_sum >> > (float_temp, ImgEst, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "float_temp_sum Error");
		
		////7、Tmp=mean(   ImgEst(:)   );
		thrust::device_ptr<float> dev_ptr(ImgEst);
		float Tmp = thrust::reduce(dev_ptr, dev_ptr + size_t(PSF_size_1*PSF_size_2), (float)0, thrust::plus<float>()) / (PSF_size_1*PSF_size_2);
		/**********************************************************************************************************/
		/*----以上正确，Tmp有误差，matlab中是47424472，C语言是47424477.675621979，差别非常非常小，应该可以忽略
		第二遍matlab是51785136，C语言是51785130.147748277，差距也非常小，可以忽略----*/
		/**********************************************************************************************************/

		////8、Ratio(1:end,1:end)=ImgExp(1:end,1:end)./(ImgEst(1:end,1:end)+Tmp/SNR)，并转成复数矩阵，虚部为零;
		Ratio_Complex_ge << <blockNum_12, threadNum_12 >> > (ImgExp, ImgEst, Tmp, SNR, Ratio_Complex, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Ratio_Complex_ge Error");
		//cufftComplex *Ratio_Complex_cpu = new cufftComplex[PSF_size_1*PSF_size_2]();
		//check(cudaMemcpy(Ratio_Complex_cpu, Ratio_Complex, sizeof(cufftComplex) * PSF_size_1*PSF_size_2, cudaMemcpyDeviceToHost), "Ratio_Complex_cpu cudaMemcpy Error");
		//float Ratio_Complex_cpu_real_sum = 0; float Ratio_Complex_cpu_imag_sum = 0;
		//for (int j = 0; j < PSF_size_1*PSF_size_2; j++)
		//{
		//	Ratio_Complex_cpu_real_sum += (Ratio_Complex_cpu[j].x);
		//	Ratio_Complex_cpu_imag_sum += (Ratio_Complex_cpu[j].y);
		//}
		//cout << "实部和：" << fixed << Ratio_Complex_cpu_real_sum << " 虚部和：" << fixed << Ratio_Complex_cpu_imag_sum << endl;

		/*******************************************************************************************/
		/*----以上正确，数组的和，matlab是0.3017935，C语言是0.301793，第二遍正确-------------------*/
		/*******************************************************************************************/

		////9、fft2(Ratio)
		res = cufftExecC2C(plan, Ratio_Complex, Ratio_Complex, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS)
		{
			cout << "Ratio_Complex cufftExecC2C error:" << res << endl;
			system("pause");
			return;
		}
		//cufftComplex *Ratio_Complex_cpu = new cufftComplex[PSF_size_1*PSF_size_2];
		//check(cudaMemcpy(Ratio_Complex_cpu, Ratio_Complex, sizeof(cufftComplex) * PSF_size_1*PSF_size_2, cudaMemcpyDeviceToHost), "Ratio_Complex_cpu cudaMemcpy Error");
		////求每一行的实部、虚部和、所有的和
		//float Ratio_Complex_cpu_realSUM = 0; float Ratio_Complex_cpu_imagSUM = 0;
		//float *abc_real_lineSUM = new float[PSF_size_1]();
		//float *abc_imag_lineSUM = new float[PSF_size_1]();
		//for (int i = 0; i < PSF_size_1; i++)
		//{
		//	for (int j = 0; j < PSF_size_2; j++)
		//	{
		//		abc_real_lineSUM[i] += Ratio_Complex_cpu[i*PSF_size_2 + j].x;
		//		abc_imag_lineSUM[i] += Ratio_Complex_cpu[i*PSF_size_2 + j].y;
		//		Ratio_Complex_cpu_realSUM += abs(Ratio_Complex_cpu[i*PSF_size_2 + j].x);
		//		Ratio_Complex_cpu_imagSUM += abs(Ratio_Complex_cpu[i*PSF_size_2 + j].y);
		//	}
		//}
		//cout << "实部和：" << fixed << Ratio_Complex_cpu_realSUM << " 虚部和：" << fixed << Ratio_Complex_cpu_imagSUM << endl;

		/*******************************************************************************************/
		/*----以上正确，数组的和和matlab不一样，可能是小数点后面位数太多了，绝对值的和是一样的-------*/
		/*******************************************************************************************/

		////10、repmat，赋值Nz遍，Ratio_Complex变成三维的fftRatio
		fftRatio_ge << <grid, block >> > (Ratio_Complex, fftRatio, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_ge Error");
		//cufftComplex *fftRatio_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(fftRatio_cpu, fftRatio, sizeof(cufftComplex) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "fftRatio_cpu cudaMemcpy Error");
		//float aaa_sum = 0; float bbb_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(fftRatio_cpu[j].x);
		//		bbb_sum += abs(fftRatio_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部绝对值和：" << fixed << aaa_sum << "虚部绝对值和：" << fixed << bbb_sum <<endl;
		//}

		////11、fftRatio.*conj(OTF)，存到fftRatio中
		fftRatio_mul_conjOTF << <blockNum_123, threadNum_123 >> > (fftRatio, OTF, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_mul_conjOTF Error");
		//cufftComplex *fftRatio_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3]();
		//check(cudaMemcpy(fftRatio_cpu, fftRatio, sizeof(cufftComplex)*PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "fftRatio_cpu cudaMemcpy Error");
		//float fftRatio_cpu_real_sum = 0; float fftRatio_cpu_imag_sum = 0;
		//float aaa_sum = 0; float bbb_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(fftRatio_cpu[j].x);
		//		bbb_sum += abs(fftRatio_cpu[j].y);
		//		fftRatio_cpu_real_sum += abs(fftRatio_cpu[j].x);
		//		fftRatio_cpu_imag_sum += abs(fftRatio_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部绝对值和：" << fixed << aaa_sum << "虚部绝对值和：" << fixed << bbb_sum <<endl;
		//}
		//cout << "实部和：" << fixed << fftRatio_cpu_real_sum << " 虚部和：" << fixed << fftRatio_cpu_imag_sum << endl;


		////12、ifft2(       fftRatio.*conj(OTF)       )，并除以总像素个数
		cufftExecC2C(fftplanfwd, fftRatio, fftRatio, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio cufftExecC2C Error");
		ifft2_divide << <blockNum_123, threadNum_123 >> > (fftRatio, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio ifft2_divide Error");
		//cufftComplex *fftRatio_cpu = new cufftComplex[PSF_size_1*PSF_size_2*PSF_size_3]();
		//check(cudaMemcpy(fftRatio_cpu, fftRatio, sizeof(cufftComplex)*PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "fftRatio_cpu cudaMemcpy Error");
		//float fftRatio_cpu_real_sum = 0; float fftRatio_cpu_imag_sum = 0;
		//float aaa_sum = 0; float bbb_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	aaa_sum = 0; bbb_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		aaa_sum += abs(fftRatio_cpu[j].x);
		//		bbb_sum += abs(fftRatio_cpu[j].y);
		//		fftRatio_cpu_real_sum += abs(fftRatio_cpu[j].x);
		//		fftRatio_cpu_imag_sum += abs(fftRatio_cpu[j].y);
		//	}
		//	cout << "第" << i + 1 << "个波段的实部绝对值和：" << fixed << aaa_sum << "虚部绝对值和：" << fixed << bbb_sum <<endl;
		//}
		//cout << "实部和：" << fixed << fftRatio_cpu_real_sum << " 虚部和：" << fixed << fftRatio_cpu_imag_sum << endl;

		////13、max(   real(   ifftshift(   ifftshift(     1),   2)   ),   0);
		ifftshift_real_max << <grid, block >> > (fftRatio, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "13、fftRatio ifftshift_real_max Error");
		////查看正确性
		//float *float_temp_cpu = new float[PSF_size_1*PSF_size_2*PSF_size_3];
		//check(cudaMemcpy(float_temp_cpu, float_temp, sizeof(float) * PSF_size_1*PSF_size_2*PSF_size_3, cudaMemcpyDeviceToHost), "float_temp_cpu cudaMemcpy Error");
		//float float_temp_cpu_sum = 0;
		//for (int i = 0; i < PSF_size_3; i++)
		//{
		//	float_temp_cpu_sum = 0;
		//	for (int j = i*PSF_size_1*PSF_size_2; j < (i + 1)*PSF_size_1*PSF_size_2; j++)
		//	{
		//		float_temp_cpu_sum += abs(float_temp_cpu[j]);
		//	}
		//	cout << "第" << i + 1 << "个波段的绝对值和：" << fixed << float_temp_cpu_sum << endl;
		//}

		////14、gpuObjRecon = gpuObjRecon.*max(  )
		real_multiply << <blockNum_123, threadNum_123 >> > (gpuObjRecon, float_temp, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon real_multiply Error");

		//cout << "完成第" << i << "次循环" << endl << endl << endl;
	}
	//计算完毕，取值放在PSF_1
	check(cudaMemcpy(PSF_1, gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3*sizeof(float), cudaMemcpyDeviceToHost), "gpuObjRecon to PSF_1 cudaMemcpy Error");
	//float PSF_1_sum = 0;
	//for (int i = 0; i < PSF_size_1*PSF_size_2*PSF_size_3; i++)
	//{
	//	PSF_1_sum += PSF_1[i];
	//}
	//cout << "gpuObjRecon和：" << fixed << PSF_1_sum << endl;

	////matlab中是从157—356行，总共356-127+1=200行。157-356列，总共356-127+1=200列。
	int line_start = Nxy / 2 - ROISize; int line_end = Nxy / 2 + ROISize - 1; int line_total = line_end - line_start + 1;
	int col_start = Nxy / 2 - ROISize; 	int col_end = Nxy / 2 + ROISize - 1; int col_total = col_end - col_start + 1;
	//float *ObjRecon = new float[line_total*col_total*PSF_size_3]();
	//float ObjRecon_sum = 0;
	//for (int band = 0; band < PSF_size_3; band++)
	//{
	//	for (int i = 0; i < line_total; i++)//行循环
	//	{
	//		for (int j = 0; j < col_total; j++)//列循环
	//		{
	//			ObjRecon[band*line_total*col_total + i*col_total + j] = PSF_1[band*PSF_size_1*PSF_size_2 + (i + line_start)*PSF_size_2 + j + col_start];
	//			ObjRecon_sum += abs(ObjRecon[band*line_total*col_total + i*col_total + j]);
	//		}
	//	}
	//}
	//cout << "ObjRecon和：" << fixed << ObjRecon_sum << endl;


	//输出图像
	GDALDriver * pDriver = GetGDALDriverManager()->GetDriverByName("ENVI");
	GDALDataset *ds = pDriver->Create(OutFile, col_total, line_total, PSF_size_3, GDT_Float32, NULL);
	if (ds == NULL)
	{
		cout << "Failed to create output file!" << endl;
		system("pause");
		return 0;
	}
	float *ObjRecon_buffer = new float[col_total];
	for (int band = 0; band < PSF_size_3; band++)
	{
		for (int i = 0; i < line_total; i++)//行循环
		{
			for (int j = 0; j < col_total; j++)//列循环
			{
				ObjRecon_buffer[j] = PSF_1[band*PSF_size_1*PSF_size_2 + (i + line_start)*PSF_size_2 + j + col_start];
			}
			ds->GetRasterBand(band + 1)->RasterIO(GF_Write, 0, i, col_total, 1, ObjRecon_buffer, col_total, 1, GDT_Float32, 0, 0);
		}
	}
	
	auto time_end = system_clock::now();
	auto duration1 = duration_cast<microseconds>(time_end - time_1);
	float usetime1 = float(duration1.count()) * microseconds::period::num / microseconds::period::den;
	auto duration2 = duration_cast<microseconds>(time_end - time_start);
	float usetime2 = float(duration2.count()) * microseconds::period::num / microseconds::period::den;
	cout << "除去数据读取和内存显存申请，数据运算部分用时：" << usetime1 << "秒" << endl;
	cout << "总用时：" << usetime2 << "秒" << endl;
	system("pause");
    return 0;
}






//int main(void)
//{
//	using namespace std;
//#define CHANNEL_NUM  31 //通道数、FFT次数
//	const int dataH = 32; //图像高度
//	const int dataW = 8;  //图像宽度
//	cufftHandle fftplanfwd;//创建句柄
//	/* 开辟主机端、设备端的内存空间 */
//	cufftComplex *h_Data = (cufftComplex*)malloc(dataH*CHANNEL_NUM*dataW * sizeof(cufftComplex));
//	cufftComplex *d_Data;//device表示GPU内存，存储从cpu拷贝到GPU的数据
//	cufftComplex *fd_Data;//device表示GPU内存,R2C后存入cufftComplex类型数据
//	cudaMalloc((void**)&d_Data, dataH*dataW*CHANNEL_NUM * sizeof(cufftComplex));
//	cudaMalloc((void**)&fd_Data, dataH*dataW*CHANNEL_NUM * sizeof(cufftComplex));
//	//随机初始化测试数据
//	for (int i = 0; i < dataH*CHANNEL_NUM; i++)
//	{
//		for (int j = 0; j < dataW; j++)
//		{
//			h_Data[i*dataW + j].x = float(rand() % 255);
//			h_Data[i*dataW + j].y = float(rand() % 255);
//		}
//	}
//	const int rank = 2;//维数
//	int n[rank] = { 32, 8 };//n*m
//	int *inembed = n;//输入的数组size
//	int istride = 1;//数组内数据连续，为1
//	int idist = n[0] * n[1];//1个数组的内存大小
//	int *onembed = n;//输出是一个数组的size
//	int ostride = 1;//每点DFT后数据连续则为1
//	int odist = n[0] * n[1];//输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
//	int batch = CHANNEL_NUM;//批量处理的批数
//	cufftPlanMany(&fftplanfwd, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);//针对多信号同时进行FFT
//	cudaMemcpy(d_Data, h_Data, dataW * dataH*CHANNEL_NUM * sizeof(cufftComplex), cudaMemcpyHostToDevice);
//	cufftExecC2C(fftplanfwd, d_Data, fd_Data, CUFFT_FORWARD);
//	cufftComplex *h_resultFFT = (cufftComplex*)malloc(dataH*dataW*CHANNEL_NUM*sizeof(cufftComplex));
//	cudaMemcpy(h_resultFFT, fd_Data, dataW*dataH*CHANNEL_NUM * sizeof(cufftComplex), cudaMemcpyDeviceToHost);//将fft后的数据拷贝回主机
//}




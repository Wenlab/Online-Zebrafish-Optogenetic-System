#include "header.cuh"
#include <chrono>
using namespace chrono;

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("GPU Parament:\n");
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
//CUDA Initialization
bool InitCUDA()
{
	int count;
	//Get the number of Cuda-enabled devices
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//Print device information
		printDeviceProp(prop);
		//Get the clock frequency of the GPU
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
//View GPU operation status
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
	//Turn the result of the difference less than 0 into 0, greater than 0 rounded, greater than 65535 into 65535
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
		float aaa = OTF[i].x*gpuObjRecon_Complex[i].x - OTF[i].y*gpuObjRecon_Complex[i].y;//Real part Results
		float bbb = OTF[i].x*gpuObjRecon_Complex[i].y + OTF[i].y*gpuObjRecon_Complex[i].x;//Virtual Part Results
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
		//Implement the image ifftshift+real+max, i.e.: divide the image into 4 quadrants, the first and third translation swap, the second and fourth translation swap
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
		//Implement the image ifftshift, i.e.: divide the image into 4 quadrants, first and third translation swap, second and fourth translation swap
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
		float aaa = fftRatio[i].x*OTF[i].x + fftRatio[i].y*OTF[i].y;//Result of real part of conjugate multiplication
		float bbb = -fftRatio[i].x*OTF[i].y + fftRatio[i].y*OTF[i].x;//Result of the virtual part of the conjugate multiplication
		fftRatio[i].x = aaa;
		fftRatio[i].y = bbb;
	}
}


int main()
{

	//Start Timer
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

	//Read tif using GDAL, using matlab resampled data
	GDALAllRegister(); OGRRegisterAll();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("SHAPE_ENCODING", "");
	GDALDataset* poSrcDS = (GDALDataset*)GDALOpen(X31_file, GA_ReadOnly);  
	if (poSrcDS == NULL)
	{
		cout << "X31_file open failed!" << endl;
		return;
	}
	int wheight = poSrcDS->GetRasterYSize();
	int wwidth = poSrcDS->GetRasterXSize();
	int bandNum = poSrcDS->GetRasterCount();
	GDALDataType dataType = poSrcDS->GetRasterBand(1)->GetRasterDataType();
	unsigned short *Img = new unsigned short[PSF_size_1*PSF_size_2]();
	for (int i = 0; i < bandNum; i++)
	{
		poSrcDS->GetRasterBand(i + 1)->RasterIO(GF_Read, 0, 0, wwidth, wheight, Img, PSF_size_1, PSF_size_2, dataType, 0, 0);
	}
	GDALClose(poSrcDS);

	/*-----------------Preparation----------------*/
	const int rank = 2;
	int n[rank] = { PSF_size_1, PSF_size_2 };//n*m
	int *inembed = n;//The size of the input array
	int istride = 1;//The data in the array is continuous is 1
	int idist = n[0] * n[1];//Memory size of one array
	int *onembed = n;//The output is the size of an array
	int ostride = 1;//1 if the data is continuous after each point of DFT
	int odist = n[0] * n[1];//Output the distance between the first array and the second array, i.e. the distance between the first elements of the two arrays
	int batch = PSF_size_3;//Number of batches
	cufftHandle fftplanfwd;//Create handle
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




	/*-------------------------------start--------------------------------------------*/
	auto time_1 = system_clock::now();
	/*-----fft2 on PSF_1-------*/
	check(cudaMemcpy(PSF_1_gpu, PSF_1, PSF_size_1*PSF_size_2*PSF_size_3*sizeof(float), cudaMemcpyHostToDevice), "PSF_1_gpu cudaMemcpy Error");
	//Convert to complex numbers with virtual part 0
	Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (PSF_1_gpu, PSF_1_gpu_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu Zhuan_Complex_kernel Error");
	

	////*------------Bulk 2D fft using cufftPlanMany's method-------------*/
	cufftExecC2C(fftplanfwd, PSF_1_gpu_Complex, OTF, CUFFT_FORWARD);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "PSF_1_gpu_Complex cufftExecC2C Error");
	
	////ImgEst is assigned a value of 0 and Ratio is assigned a value of 1
	initial_kernel_1 << <blockNum_12, threadNum_12 >> > (ImgEst, Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_1 Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");
	////gpuObjRecROI is assigned a value of 1
	initial_kernel_3 << <blockNum_ROI, threadNum_ROI >> > (gpuObjRecROI, ROISize * 2 * ROISize * 2 * Nz);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "initial_kernel_3 Error");

	//Copy tif data to video memory
	check(cudaMemcpy(Img_gpu, Img, PSF_size_1*PSF_size_2 * sizeof(unsigned short), cudaMemcpyHostToDevice), "Img_gpu cudaMemcpy Error");
	//Subtract the background mean value and put the result in the float type array ImgExp
	ImgExp_ge << <blockNum_12, threadNum_12 >> > (Img_gpu, BkgMean, ImgExp, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "ImgExp_ge Error");

	//The elements of Ratio and gpuObjRecon are assigned the value 1
	Ratio_fuzhi << <blockNum_12, threadNum_12 >> > (Ratio, PSF_size_1*PSF_size_2);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "Ratio_fuzhi Error");
	gpuObjRecon_fuzhi << <blockNum_123, threadNum_123 >> > (gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3);
	cudaDeviceSynchronize();
	checkGPUStatus(cudaGetLastError(), "gpuObjRecon_fuzhi Error");


	//iteration
	for (int i = 0; i < ItN; i++)
	{
		////1 fft2(gpuObjRecon)
		Zhuan_Complex_kernel << <blockNum_123, threadNum_123 >> > (gpuObjRecon, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Zhuan_Complex_kernel Error");
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C Error");

		////2 OTF.*fft2(gpuObjRecon_Complex),The results are placed in gpuObjRecon_Complex
		OTF_mul_gpuObjRecon_Complex << <blockNum_123, threadNum_123 >> > (OTF, gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "2、OTF.*fft2(gpuObjRecon_Complex) Error");

		////3 ifft2(OTF.*fft2(gpuObjRecon)) The inverse conversion requires dividing by the total number of pixels
		cufftExecC2C(fftplanfwd, gpuObjRecon_Complex, gpuObjRecon_Complex, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex cufftExecC2C cufft_inverse Error");
		////4 Divide by the total number of pixels to be correct
		ifft2_divide << <blockNum_123, threadNum_123 >> > (gpuObjRecon_Complex, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon_Complex ifft2_divide Error");

		////5 ifftshift + real + max(,0), Get the real part of the matrix float_temp, less than 0 assign 0
		ifftshift_real_max << <grid, block >> > (gpuObjRecon_Complex, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "5、gpuObjRecon_Complex ifftshift_real_max Error");

		////6 sum( ,3), Calculate the sum in the third dimension and return the matrix ImgEst of PSF_size_1 rows and PSF_size_2 columns
		float_temp_sum << <grid_sum, block_sum >> > (float_temp, ImgEst, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "float_temp_sum Error");
		
		////7、Tmp=mean(   ImgEst(:)   );
		thrust::device_ptr<float> dev_ptr(ImgEst);
		float Tmp = thrust::reduce(dev_ptr, dev_ptr + size_t(PSF_size_1*PSF_size_2), (float)0, thrust::plus<float>()) / (PSF_size_1*PSF_size_2);


		////8、Ratio(1:end,1:end)=ImgExp(1:end,1:end)./(ImgEst(1:end,1:end)+Tmp/SNR),and transformed into a complex matrix with zero virtual part;
		Ratio_Complex_ge << <blockNum_12, threadNum_12 >> > (ImgExp, ImgEst, Tmp, SNR, Ratio_Complex, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "Ratio_Complex_ge Error");

		////9、fft2(Ratio)
		res = cufftExecC2C(plan, Ratio_Complex, Ratio_Complex, CUFFT_FORWARD);
		if (res != CUFFT_SUCCESS)
		{
			cout << "Ratio_Complex cufftExecC2C error:" << res << endl;
			system("pause");
			return;
		}

		////10、repmat,Assign Nz times, Ratio_Complex becomes a three-dimensional fftRatio
		fftRatio_ge << <grid, block >> > (Ratio_Complex, fftRatio, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_ge Error");

		////11、fftRatio.*conj(OTF), Save to fftRatio
		fftRatio_mul_conjOTF << <blockNum_123, threadNum_123 >> > (fftRatio, OTF, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio_mul_conjOTF Error");

		////12、ifft2(       fftRatio.*conj(OTF)       ), Divided by the total number of pixels
		cufftExecC2C(fftplanfwd, fftRatio, fftRatio, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio cufftExecC2C Error");
		ifft2_divide << <blockNum_123, threadNum_123 >> > (fftRatio, PSF_size_1*PSF_size_2*PSF_size_3, PSF_size_1*PSF_size_2);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "fftRatio ifft2_divide Error");

		////13、max(   real(   ifftshift(   ifftshift(     1),   2)   ),   0);
		ifftshift_real_max << <grid, block >> > (fftRatio, float_temp, PSF_size_1, PSF_size_2, PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "13、fftRatio ifftshift_real_max Error");

		////14、gpuObjRecon = gpuObjRecon.*max(  )
		real_multiply << <blockNum_123, threadNum_123 >> > (gpuObjRecon, float_temp, PSF_size_1*PSF_size_2*PSF_size_3);
		cudaDeviceSynchronize();
		checkGPUStatus(cudaGetLastError(), "gpuObjRecon real_multiply Error");

		//cout <<  i <<  endl << endl << endl;
	}
	//The calculation is completed and the value is placed in PSF_1
	check(cudaMemcpy(PSF_1, gpuObjRecon, PSF_size_1*PSF_size_2*PSF_size_3*sizeof(float), cudaMemcpyDeviceToHost), "gpuObjRecon to PSF_1 cudaMemcpy Error");

	////matlab is from 157-356 rows, total 356-127+1=200 rows. 157-356 columns, total 356-127+1=200 columns
	int line_start = Nxy / 2 - ROISize; int line_end = Nxy / 2 + ROISize - 1; int line_total = line_end - line_start + 1;
	int col_start = Nxy / 2 - ROISize; 	int col_end = Nxy / 2 + ROISize - 1; int col_total = col_end - col_start + 1;


	//output image
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
		for (int i = 0; i < line_total; i++)//row
		{
			for (int j = 0; j < col_total; j++)//col
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
	cout << "Excluding data reads and memory memory requests, the data computation part takes time:" << usetime1 << "second" << endl;
	cout << "total time:" << usetime2 << "second" << endl;
	system("pause");
    return 0;
}




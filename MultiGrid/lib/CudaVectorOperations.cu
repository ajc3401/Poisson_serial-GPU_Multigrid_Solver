#include "CudaVectorOperations.cuh"
#include "CudaKernels.cu"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cfloat>
#include "CudaMemoryHandler.cuh"
#include <iostream>
#include <cublas_v2.h>

template<class T> void setEqual(T* a, const T* b, size_t N)
{

	SetEqual << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
	cudaDeviceSynchronize();

};

void applyFunction(double* a, size_t N, std::string function)
{
	if (function == std::string("sin"))
		SinElements << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, N);
	else
		std::runtime_error("Only sin supported");
	cudaDeviceSynchronize();
}

void applyFunction(float* a, size_t N, std::string function)
{
	if (function == std::string("sin"))
		SinElements << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, N);
	else
		std::runtime_error("Only sin supported");
	cudaDeviceSynchronize();
}

void applyFunction(int* a, size_t N, std::string function)
{
	
	std::runtime_error("Int not supported for applyFunction");
}

void applyFunction(size_t* a, size_t N, std::string function)
{

	std::runtime_error("size_t not supported for applyFunction");
}

template<class T> void setNegative(T* a, size_t N)
{
	SetNegative << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, N);
	cudaDeviceSynchronize();
}

template<class T> void sumVectors(T* a, const T* b, size_t N)
{
	
	VectorAdd << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
	cudaDeviceSynchronize();
	
};

template<class T> void subtractVectors(T* a, const T* b, size_t N)
{

	VectorSubtract << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
	cudaDeviceSynchronize();

};

template<class T> void scalarVectorMultiply(T* a, const T b, size_t N)
{
	VectorScalarMultiply << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
	cudaDeviceSynchronize();
}

void l2Norm(float* result, const float* invec, size_t N)
{
	cublasHandle_t handle = NULL;
	cublasCreate(&handle);

	cublasSnrm2(handle, N, invec, 1, result);

	cudaDeviceSynchronize();
}

void l2Norm(double* result, const double* invec, size_t N)
{
	cublasHandle_t handle = NULL;
	cublasCreate(&handle);

	cublasDnrm2(handle, N, invec, 1, result);

	cudaDeviceSynchronize();
}

void l2Norm(int* result, const int* invec, size_t N)
{
	std::runtime_error("Integer l2 norm not currently supported.");
}

void l2Norm(size_t* result, const size_t* invec, size_t N)
{
	std::runtime_error("Size_t l2 norm not currently supported.");
}

template<class T> void interpolate1D(T* v_finer, const T* v_coarser, size_t N_coarser)
{
	Interpolate1D << <(N_coarser - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (v_finer, v_coarser, N_coarser);
	cudaDeviceSynchronize();
	//std::cout << "v_finer = " << v_finer[0];
}

template<class T> void interject1D(T* v_coarser, const T* v_finer, size_t N_coarser)
{
	Interject1D << <(N_coarser - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (v_coarser, v_finer, N_coarser);
	cudaDeviceSynchronize();
	std::cout << "v_finer = " << v_finer[14];
}

template<class T> void invertElements(T* a, size_t N)
{
	InvertElements << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, N);
	cudaDeviceSynchronize();
}

template<class T> void dotProduct(const T* a, const T* b, T& c, size_t N)
{
	T* tmp{ CudaMemoryHandler<T>::allocate((N - 1) / BLOCK_SIZE + 1) };

	VectorDotProduct << < (N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, tmp, N);

	cudaDeviceSynchronize();

	c = 0;
	//std::accumulate(tmp, tmp + ((N - 1) / BLOCK_SIZE + 1), c);
	for (size_t i = 0; i < ((N - 1) / BLOCK_SIZE + 1); i++)
	{
		c += tmp[i];
	}
	/*for (auto x : tmp)
	{
		c += x;
	}*/
	CudaMemoryHandler<T>::deallocate(tmp);
} 

template<class T> void setValue(T* a, const T b, size_t N)
{
	
	SetValue << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
	
	cudaDeviceSynchronize();
	
}

template<class T> void setRange(size_t left, size_t right, T* a, const T* b)
{
	size_t N{ right - left };

	std::cout << "GPU Set Range" << std::endl;
	
	SetRange << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (left, a, b, N);

	cudaDeviceSynchronize();
}
template void setEqual(double* a, const double* b, size_t N);
template void setEqual(size_t* a, const size_t* b, size_t N);
template void setEqual(int* a, const int* b, size_t N);
template void setEqual(float* a, const float* b, size_t N);

template void setNegative(double* a, size_t N);
template void setNegative(size_t* a, size_t N);
template void setNegative(int* a, size_t N);
template void setNegative(float* a, size_t N);

//template void applyFunction(double* a, size_t N, std::string function);
//template void applyFunction(size_t* a, size_t N, std::string function);
//template void applyFunction(int* a, size_t N, std::string function);
//template void applyFunction(float* a, size_t N, std::string function);

template void sumVectors(double* a, const double* b, size_t N);
template void sumVectors(size_t* a, const size_t* b, size_t N);
template void sumVectors(int* a, const int* b, size_t N);
template void sumVectors(float* a, const float* b, size_t N);

template void scalarVectorMultiply(double* a, const double b, size_t N);
template void scalarVectorMultiply(size_t* a, const size_t b, size_t N);
template void scalarVectorMultiply(int* a, const int b, size_t N);
template void scalarVectorMultiply(float* a, const float b, size_t N);

template void invertElements(double* a, size_t N);
template void invertElements(size_t* a, size_t N);
template void invertElements(float* a, size_t N);
template void invertElements(int* a, size_t N);

template void interpolate1D(double* v_finer, const double* v_coarser, size_t N_coarser);
template void interpolate1D(size_t* v_finer, const size_t* v_coarser, size_t N_coarser);
template void interpolate1D(float* v_finer, const float* v_coarser, size_t N_coarser);
template void interpolate1D(int* v_finer, const int* v_coarser, size_t N_coarser);

template void interject1D(double* v_coarser, const double* v_finer, size_t N_coarser);
template void interject1D(size_t* v_coarser, const size_t* v_finer, size_t N_coarser);
template void interject1D(float* v_coarser, const float* v_finer, size_t N_coarser);
template void interject1D(int* v_coarser, const int* v_finer, size_t N_coarser);


template void subtractVectors(double* a, const double* b, size_t N);
template void subtractVectors(int* a, const int* b, size_t N);
template void subtractVectors(size_t* a, const size_t* b, size_t N);
template void subtractVectors(float* a, const float* b, size_t N);

template void dotProduct(const double* a, const double* b, double& c, size_t N);
template void dotProduct(const int* a, const int* b, int& c, size_t N);
template void dotProduct(const size_t* a, const size_t* b, size_t& c, size_t N);
template void dotProduct(const float* a, const float* b, float& c, size_t N);

template void setValue(double* a, const double b, size_t N);
template void setValue(int* a, const int b, size_t N);
template void setValue(float* a, const float b, size_t N);
template void setValue(size_t* a, const size_t b, size_t N);

template void setRange(size_t left, size_t right, double* a, const double* b);
template void setRange(size_t left, size_t right, int* a, const int* b);
template void setRange(size_t left, size_t right, float* a, const float* b);
template void setRange(size_t left, size_t right, size_t* a, const size_t* b);
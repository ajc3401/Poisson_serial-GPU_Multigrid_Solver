#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


#define BLOCK_SIZE 32


template<class T>
__global__ void VectorAdd(T* a, const T* b, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] += b[tid];
	}

}

template<class T>
__global__ void VectorSubtract(T* a, const T* b, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] -= b[tid];
	}

}

template<class T>
__global__ void VectorScalarMultiply(T* a, const T b, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] *= b;
	}

}

template<class T>
__global__ void VectorDotProduct(const T* a, const T* b, T* c, size_t N)
{
	__shared__ T cache[BLOCK_SIZE];
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t cacheIdx = threadIdx.x;

	T tmp{ 0 };
	if (tid < N) {
		tmp += a[tid] * b[tid];
	}

	cache[cacheIdx] = tmp;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIdx < i)
			cache[cacheIdx] += cache[cacheIdx + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIdx == 0)
		c[blockIdx.x] = cache[0];
}

template<class T>
__global__ void SinElements(T* a, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		T val{ a[tid] };
		a[tid] = sin(val);
	}

}

template<class T>
__global__ void InvertElements(T* a, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if ((tid < N) && (a[tid] > 0)) {
		a[tid] = 1/a[tid];
	}

}

template<class T>
__global__ void SetValue(T* a, const T b, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] = b;
	}

}

template<class T>
__global__ void SetEqual(T* a, const T* b, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] = b[tid];
	}

}

template<class T>
__global__ void SetNegative(T* a, size_t N)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] = -a[tid];
	}

}

template<class T>
__global__ void SetDiagonal(T* a, const T b, size_t offset, size_t M, size_t N)
{
	// Assume MxN matrix
	size_t row = threadIdx.x + blockIdx.x * blockDim.x;

	
	if (row < M)
		a[row + offset + M * row] = b;
	
	//else
	//{
	//	//if ((row < M) && (col < N))
	//	
	//	if(row < M)
	//		a[row + M * (row - offset)] = b;
	//}

}

// Assumes N = right - left
template<class T>
__global__ void SetRange(size_t left, T* a, const T* b, size_t N)
{
	// Assume MxN matrix
	//size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) {
		a[tid] = b[tid+left];
	}
	

}

template<class T>
__global__ void GetDiagonal(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	// Assume MxN matrix
	//size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.x + blockIdx.x * blockDim.x;

	
	if (row < M)
		a[row + offset + M * row] = b[row + offset + M * row];
	//}
	//else
	//{
	//	//if ((row < M) && (col < N))
	//	if (row < M)
	//		a[row + M * (row - offset)] = b[row + M * (row - offset)];
	//}

}

template<class T>
__global__ void Interpolate1D(T* v_finer, const T* v_coarser, size_t N_coarser)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	T const c0{ 0.5 };
	if (tid < N_coarser)
	{
		v_finer[2 * tid + 1] = v_coarser[tid];
		v_finer[2 * (tid + 1)] = c0 * (v_coarser[tid] + v_coarser[tid+1]);
	}

	
}

template<class T>
__global__ void Interject1D(T* v_coarser, const T* v_finer, size_t N_coarser)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	T const c0{ 0.25 };
	T const c1{ 2.0 };
	if (1 <= tid < N_coarser - 1)
	{
		v_coarser[tid] = c0 * (v_finer[2*tid - 1] + c1*v_finer[2 * tid] + v_finer[2 * tid + 1]);
		
	}


}

template<class T>
__global__ void GetLowerTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	// Assume MxN matrix
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.y + blockIdx.y * blockDim.y;

	if ((row < M) && (col < row))
		a[row + M * col] = b[row + M * col];
	//if (((row < M) && (row >= (col + 1 + offset))) && (col < N))
		//a[row + M * col] = b[row + M * col];
	
}

template<class T>
__global__ void GetUpperTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	// Assume MxN matrix
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	size_t row = threadIdx.y + blockIdx.y * blockDim.y;

	if ((col < N) && (row < col) && (col >= (1 + offset)))
		a[row + M * col] = b[row + M * col];
		
	
	

}


//template <class T>
//__global__ void MatrixMul(const T* a, const T* b, T* c, size_t sharedmem_size)
//{
//	__shared__ T A[sharedmem_size];
//	__shared__ T B[sharedmem_size];
//
//	
//
//	size_t ROW = blocokIdx.y * tile_size + threadIdx.y;
//	size_t COL = blockIdx.x * tile_size + threadIdx.x;
//
//	T tmp{ 0 };
//
//	for (size_t i = 0; i < (N / tile_size); i++)
//	{
//		A[(ty * tile_size) + tx] = a[ROW * N + (i * tile_size)]
//	}
//}
template __global__ void SetEqual(double* a, const double* b, size_t N);
template __global__ void SetEqual(float* a, const float* b, size_t N);

template __global__ void VectorAdd(double* a, const double* b, size_t N);
template __global__ void VectorAdd(float* a, const float* b, size_t N);

template __global__ void VectorSubtract(double* a, const double* b, size_t N);
template __global__ void VectorSubtract(float* a, const float* b, size_t N);

template __global__ void VectorScalarMultiply(double* a, const double b, size_t N);
template __global__ void VectorScalarMultiply(float* a, const float b, size_t N);

template __global__ void InvertElements(double* a, size_t N);
template __global__ void InvertElements(float* a, size_t N);

template __global__ void SinElements(double* a, size_t N);
template __global__ void SinElements(float* a, size_t N);

template __global__ void Interpolate1D(double* v_finer, const double* v_coarser, size_t N_coarser);
template __global__ void Interpolate1D(float* v_finer, const float* v_coarser, size_t N_coarser);

template __global__ void Interject1D(double* v_coarser, const double* v_finer, size_t N_coarser);
template __global__ void Interject1D(float* v_coarser, const float* v_finer, size_t N_coarser);

template __global__ void VectorDotProduct(const double* a, const double* b, double* c, size_t N);
template __global__ void VectorDotProduct(const float* a, const float* b, float* c, size_t N);

template __global__ void SetDiagonal(double* a, const double b, size_t offset, size_t M, size_t N);
template __global__ void SetDiagonal(float* a, const float b, size_t offset, size_t M, size_t N);

template __global__ void SetRange(size_t left, double* a, const double* b, size_t N);
template __global__ void SetRange(size_t left, float* a, const float* b, size_t N);

template __global__ void GetDiagonal(double* a, const double* b, size_t offset, size_t M, size_t N);
template __global__ void GetDiagonal(float* a, const float* b, size_t offset, size_t M, size_t N);

template __global__ void GetLowerTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template __global__ void GetLowerTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);

template __global__ void GetUpperTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template __global__ void GetUpperTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);

//template __global__ void GetUpperTriangular(double* a, const double* b, int offset, int M, int N);
//template __global__ void GetUpperTriangular(float* a, const float* b, int offset, int M, int N);
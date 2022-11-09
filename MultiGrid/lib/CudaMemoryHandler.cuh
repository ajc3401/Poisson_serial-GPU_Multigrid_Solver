// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef CUDAMEMORYHANDLER_CUH
#define CUDAMEMORYHANDLER_CUH
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "CudaErrorHandler.cuh"

template<class T>
class CudaMemoryHandler
{
public:
	using value_type = T;

	CudaMemoryHandler() noexcept {}
	template<class U> CudaMemoryHandler(CudaMemoryHandler<U> const&) noexcept {}

	static auto allocate(size_t nalloc, value_type* dev_p)
	{
		return HANDLE_ERROR(cudaMallocManaged(&dev_p, nalloc * sizeof(value_type)));
	}

	static value_type* allocate(size_t nalloc)
	{
		value_type* tmp{ nullptr };
		HANDLE_ERROR(cudaMallocManaged((void**)&tmp, nalloc * sizeof(value_type)));
		return tmp;
	}

	static void deallocate(value_type* p)
	{
		HANDLE_ERROR ( cudaFree(p) );
	}
};

template class CudaMemoryHandler<double>;
template class CudaMemoryHandler<float>;
template class CudaMemoryHandler<int>;
#endif // !

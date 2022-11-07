//#include "SerialVectorOperations.h"
//
//template<class T> void sumVectors(T* a, const T* b, size_t N)
//{
//
//	VectorAdd << <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, N);
//	cudaDeviceSynchronize();
//
//};
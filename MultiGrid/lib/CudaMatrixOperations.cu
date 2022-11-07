#include "CudaMatrixOperations.cuh"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cfloat>
#include "CudaMemoryHandler.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "CudaKernels.cu"
#include "cusparse_v2.h"

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

void matrixMultiply(const float* a, const float* b, float* c, size_t N, size_t M, size_t K)
{
	// We obtain the matrix (M X K ) = (M X N) * (N X K)
	cublasHandle_t handle = NULL;
	cublasCreate(&handle);

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, N, &beta, c, M);

	cudaDeviceSynchronize();
}

void matrixMultiply(const int* a, const int* b, int* c, size_t N, size_t M, size_t K)
{
	std::runtime_error("Integer matrix multiplication not currently supported.");
}
void matrixMultiply(const double* a, const double* b, double* c, size_t N, size_t M, size_t K)
{
	// We obtain the matrix (M X K ) = (M X N) * (N X K)
	cublasHandle_t handle;
	cublasCreate(&handle);

	const double alpha = 1.0;
	const double beta = 0.0;

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, a, M, b, N, &beta, c, M);

	cudaDeviceSynchronize();

	std::cout << "Cublas Matrix Multiplication" << std::endl;
}

// Ax = y
void sparseMatrixVectorMultiply(void* Aval, void* Arow, void* Acol, void* x, void* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ)
{
	float alpha = 1.0f;
	float beta = 0.0f;

	void* dBuffer {nullptr};
	size_t bufferSize = 0;

	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	
	CHECK_CUSPARSE(cusparseCreate(&handle));

	CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_NNZ, Arow, Acol, Aval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, x, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, y, CUDA_R_32F));
	
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_MV_ALG_DEFAULT, &bufferSize));

	HANDLE_ERROR( cudaMalloc(&dBuffer, bufferSize) );

	CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_MV_ALG_DEFAULT, dBuffer));

	HANDLE_ERROR(cudaFree(dBuffer));

	CHECK_CUSPARSE(cusparseDestroySpMat(matA));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	CHECK_CUSPARSE(cusparseDestroy(handle));

	cudaDeviceSynchronize();
}

void sparseMatrixMatrixMultiply(void* Aval, void* Arow, void* Acol, void* Bval, void* Brow, void* Bcol,
	void* Cval, void* Crow, void* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ, 
	size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ)
{
	cusparseHandle_t     handle = NULL;
	cusparseSpMatDescr_t matA, matB, matC;

	float alpha = 1.0f;
	float beta = 0.0f;

	void* dBuffer1 {nullptr}, * dBuffer2 {nullptr};
	size_t bufferSize1 = 0, bufferSize2 = 0;


	

	CHECK_CUSPARSE(cusparseCreate(&handle));
		
	CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_NNZ,
		Arow, Acol, Aval,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_NNZ,
		Brow, Bcol, Bval,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
		NULL, NULL, NULL,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	cusparseSpGEMMDescr_t spgemmDesc;
	CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));


	CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
				&alpha, matA, matB, &beta, matC,
				CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
				spgemmDesc, &bufferSize1, NULL));

	HANDLE_ERROR(cudaMalloc((void**)&dBuffer1, bufferSize1));


	CHECK_CUSPARSE(
		cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
			spgemmDesc, &bufferSize1, dBuffer1));

	CHECK_CUSPARSE(
		cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
			spgemmDesc, &bufferSize2, NULL));
	
	HANDLE_ERROR(cudaMalloc((void**)&dBuffer2, bufferSize2));

	CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, matB, &beta, matC,
		CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT,
		spgemmDesc, &bufferSize2, dBuffer2));
	
		int64_t C_num_rows1, C_num_cols1, C_nnz1;
	CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
			&C_nnz1));
	// Set the number of rows, cols, NNZ
	C_num_rows = C_num_rows1;
	C_num_cols = C_num_cols1;
	C_NNZ = C_nnz1;

	// allocate matrix C
	

	CHECK_CUSPARSE(cusparseCsrSetPointers(matC, Crow, Ccol, Cval));

	CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
			&alpha, matA, matB, &beta, matC,
			CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

	CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
	CHECK_CUSPARSE(cusparseDestroySpMat(matA));
	CHECK_CUSPARSE(cusparseDestroySpMat(matB));
	CHECK_CUSPARSE(cusparseDestroySpMat(matC));
	CHECK_CUSPARSE(cusparseDestroy(handle));

	cudaDeviceSynchronize();
	
}
void convertDensetoCSR(void* dense, size_t nrows, size_t ncols, void* csr_val, void* csr_row, void* csr_col, size_t& nNNZ)
{
	void* dBuffer {nullptr};
	size_t bufferSize = 0;
	size_t leading_dim{ nrows };
	
	cusparseHandle_t handle = NULL;
	cusparseSpMatDescr_t csr_mat;
	cusparseDnMatDescr_t dense_mat;

	CHECK_CUSPARSE(cusparseCreate(&handle));
	// Create dense matrix
	CHECK_CUSPARSE(cusparseCreateDnMat(&dense_mat, nrows, ncols, leading_dim, dense,
		CUDA_R_32F, CUSPARSE_ORDER_COL));
	// Create sparse matrix
	CHECK_CUSPARSE(cusparseCreateCsr(&csr_mat, nrows, ncols, 0,
		csr_row, NULL, NULL,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	// Create external buffer
	CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle, dense_mat, csr_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
		&bufferSize))
		

	HANDLE_ERROR(cudaMalloc(&dBuffer, bufferSize));

	// Do the actual dense to CSR conversion
	CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, dense_mat, csr_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

	int64_t nrows_tmp, ncols_tmp, NNZ;
	CHECK_CUSPARSE(cusparseSpMatGetSize(csr_mat, &nrows_tmp, &ncols_tmp, &NNZ));

	nNNZ = NNZ;

	CHECK_CUSPARSE(cusparseCsrSetPointers(csr_mat, csr_row, csr_col, csr_val));

	CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, dense_mat, csr_mat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
		

	HANDLE_ERROR(cudaFree(dBuffer));

	CHECK_CUSPARSE(cusparseDestroyDnMat(dense_mat));
	CHECK_CUSPARSE(cusparseDestroySpMat(csr_mat));
	CHECK_CUSPARSE(cusparseDestroy(handle));

	cudaDeviceSynchronize();

	
}

template <class T> void setDiagonal(T* a, const T number, size_t offset, size_t M, size_t N)
{
	size_t MATRIX_SIZE{ M * N };
	SetDiagonal << <(MATRIX_SIZE - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, number, offset, M, N);
	cudaDeviceSynchronize();
}

template <class T> void getDiagonal(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	size_t MATRIX_SIZE{ M * N };
	GetDiagonal << <(MATRIX_SIZE - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (a, b, offset, M, N);
	cudaDeviceSynchronize();
}

template <class T> void getLowerTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	
	size_t MATRIX_SIZE{ M * N };
	size_t GRID_SIZE{ (MATRIX_SIZE - 1) / BLOCK_SIZE + 1 };
	
	dim3 THREADS(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GRID(GRID_SIZE, GRID_SIZE);
	
	GetLowerTriangular << <GRID, THREADS >> > (a, b, offset, M, N);
	cudaDeviceSynchronize();
}

template <class T> void getUpperTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{
	size_t MATRIX_SIZE{ M * N };

	size_t GRID_SIZE{ (MATRIX_SIZE - 1) / BLOCK_SIZE + 1 };

	dim3 THREADS(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GRID(GRID_SIZE, GRID_SIZE);


	GetUpperTriangular << <GRID, THREADS >> > (a, b, offset, M, N);
	cudaDeviceSynchronize();
	
}

template void setDiagonal(double* a, const double number, size_t offset, size_t M, size_t N);
template void setDiagonal(float* a, const float number, size_t offset, size_t M, size_t N);

template void getDiagonal(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getDiagonal(float* a, const float* b, size_t offset, size_t M, size_t N);

template void getLowerTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getLowerTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);

template void getUpperTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getUpperTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);


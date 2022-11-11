// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef SERIALMATRIXOPERATIONS_H
#define SERIALMATRIXOPERATIONS_H
#include "VectorBase.h"
// Define all the matrix operations for the CPU implementation

// These assume a column major layout of the matrix
template <class T> void matrixMultiply(const T* a, const T* b, T* c, size_t N, size_t M, size_t K);

template <class T> void sparseMatrixMatrixMultiply(T* Aval, int* Arow, int* Acol, T* Bval, int* Brow, int* Bcol,
	T* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
	size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);

// Need pop back method for transpose so we pass in VectorBase pointers instead of the underlying pointer they point to
template <class T> void sparseMatrixTranspose(VectorBase<T>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<T>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows);

template<class T> void sparseMatrixVectorMultiply(T* Aval, int* Arow, int* Acol, T* x, T* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);

template <class T> void convertDensetoCSR(T* dense, size_t nrows, size_t ncols, VectorBase<T>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ);

// The offset is the offset from main diagonal, if it's positive then it's an offset along the rows and if negative
// then an offset along the column
template <class T> void setDiagonal(T* a, const T number, size_t offset, size_t M, size_t N);

// Get diagonal of b and sets it to diagonal of a.
template <class T> void getDiagonal(T* a, const T* b, size_t offset, size_t M, size_t N);
//Get lower trinagular form of b and set it to a.  Offset of 0 means that the diagonal is also 0'd out.
template <class T> void getLowerTriangular(T* a, const T* b, size_t offset, size_t M, size_t N);
//Get upper trinagular form of b and set it to a.  Offset of 0 means that the diagonal is also 0'd out.
template <class T> void getUpperTriangular(T* a, const T* b, size_t offset, size_t M, size_t N);
#endif

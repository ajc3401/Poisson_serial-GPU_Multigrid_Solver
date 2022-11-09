// Copyright 2022, Anthony Cooper, All rights reserved

#include "SerialMatrixOperations.h"
#include <assert.h>
#include <vector>
#include <iostream>

template <class T> void matrixMultiply(const T* a, const T* b, T* c, size_t N, size_t M, size_t K)
{
	// This is done with column major order.
    // We obtain the matrix (M X K ) = (M X N) * (N X K)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < K; j++) {
            c[j * M + i] = 0;

            for (size_t k = 0; k < N; k++) {
                c[j * M + i] += a[k * M + i] * b[j * N + k];
            }
        }
    }
}

// Dense is in column major order
template <class T> void convertDensetoCSR(T* dense, size_t nrows, size_t ncols, VectorBase<T>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ)
{
    csr_row->push_back(0);
  
    // Iterate across rows to form row and value pointer
    int counter{ 0 };
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if (dense[i + j * nrows] != 0)
            {
                csr_val->push_back(dense[i + j * nrows]);
                csr_col->push_back(i);
                counter++;
            }
        }
        csr_row->push_back(counter);
    }
    // Set NNZ to new number of elements
    nNNZ = counter;
}

template <class T> void sparseMatrixTranspose(VectorBase<T>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<T>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows)
{
    // Get the number of elements per column
    for (size_t i = 0; i < in_NNZ; ++i) 
        ++resrow->begin()[incol->begin()[i] + 2];
    
    
    // Generate shifted row pointer from element count per column
    for (size_t i = 2; i < in_num_rows+1; ++i) {
        // create incremental sum
        resrow->begin()[i] += resrow->begin()[i - 1];
    }
    //std::cout << "in_NNZ = " << in_NNZ << std::endl;
    //std::cout << "inrow begin 11 = " << inrow->begin()[in_NNZ] << std::endl;
    for (int i = 0; i < in_num_rows; ++i) 
    {
        for (size_t j = inrow->begin()[i]; j < inrow->begin()[i + 1]; ++j) {
            // calculate index to transposed matrix at which we should place current element, and at the same time build final rowPtr
            /*std::cout << "inrow at i = " << inrow->begin()[i] << std::endl;
            std::cout << "j is " << j << std::endl;
            std::cout << "i is" << i << std::endl;
            std::cout << "incol is " << incol->begin()[j] + 1 << std::endl;*/
            const size_t new_index = resrow->begin()[incol->begin()[j] + 1]++;
            resval->begin()[new_index] = inval->begin()[j];
            rescol->begin()[new_index] = i;
        }
    }
    resrow->pop_back(); // pop that one extra

}

template<class T> void sparseMatrixVectorMultiply(T* Aval, int* Arow, int* Acol, T* x, T* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ)
{
    
    for (size_t i = 0; i < A_num_cols; i++)
    {
        y[i] = 0.0;
        std::cout << "i = " << i << std::endl;
        for (size_t j = Arow[i]; j < Arow[i + 1]; j++)
        {
            //std::cout << "inrow at i = " << inrow->begin()[i] << std::endl;
            
            std::cout << "Aval at j " << Aval[j] << std::endl;
            std::cout << "x at Acolj " << x[Acol[j]] << std::endl;
            y[i] += Aval[j] * x[Acol[j]];
        }
    }
}

// Matrix is MXN
template <class T> void setDiagonal(T* a, const T number, size_t offset, size_t M, size_t N)
{
    if (offset >= 0)
    {
        for (size_t i = 0; i < M; i++) {
                a[i + offset + M * i] = number;
            }
        
    }
    else
    {
        for (size_t i = 0; i < M; i++) {
            
                a[i + M * (i - offset)] = number;
            
        }
    }

}
// Matrix is MXN
template <class T> void getDiagonal(T* a, const T* b, size_t offset, size_t M, size_t N)
{
    if (offset >= 0)
    {
        for (size_t i = 0; i < M; i++) {
            a[i + offset + M * i] = b[i + offset + M * i];
        }

    }
    else
    {
        for (size_t i = 0; i < M; i++) {

            a[i + M * (i - offset)] = b[i + M * (i - offset)];

        }
    }
}
// Matrix is MXN
template <class T> void getLowerTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{
   
  for (size_t i = 0; i < N; i++) {
      for (size_t j = (i + 1 + offset); j < M; j++) {
                a[j + M * i] = b[j + M * i];
      }
  }
        
}

// Matrix is MXN
template <class T> void getUpperTriangular(T* a, const T* b, size_t offset, size_t M, size_t N)
{

    for (size_t i = (1 + offset); i < N; i++) {
        for (size_t j = 0; ((j < M) || (j<i)); j++) {
            a[j + M * i] = b[j + M * i];
        }
    }

}

template <class T> void sparseMatrixMatrixMultiply(T* Aval, int* Arow, int* Acol, T* Bval, int* Brow, int* Bcol,
    T* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
    size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ)
{
    assert(A_num_cols == B_num_rows);
    Crow[0] = 0;
    for (size_t i = 0; i < A_num_rows; i++)
    {
        // If no elements in this row then continue
        if (Arow[i] == Arow[i + 1])
        {
            Crow[i + 1] = 0;
            continue;
        }


        std::vector<int> curr_vals_A{};

        for (int iA = Arow[i]; iA < Arow[i + 1]; iA++)
        {
            curr_vals_A.push_back(Aval[iA]);
        }
        for (size_t j = 0; j < B_num_cols; j++)
        {
            std::vector<int> curr_vals_B{};



            for (int jB = 0; jB <= B_NNZ; jB++)
            {
                if (Bcol[jB] == j)
                    curr_vals_B.push_back(Bval[jB]);

            }


        }
    }
}
template void matrixMultiply(const double* a, const double* b, double* c, size_t N, size_t M, size_t K);
template void matrixMultiply(const float* a, const float* b, float* c, size_t N, size_t M, size_t K);
template void matrixMultiply(const int* a, const int* b, int* c, size_t N, size_t M, size_t K);

template void convertDensetoCSR(float* dense, size_t nrows, size_t ncols, VectorBase<float>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ);
template void convertDensetoCSR(double* dense, size_t nrows, size_t ncols, VectorBase<double>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ);
template void convertDensetoCSR(int* dense, size_t nrows, size_t ncols, VectorBase<int>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ);
template void convertDensetoCSR(size_t* dense, size_t nrows, size_t ncols, VectorBase<size_t>* csr_val, VectorBase<int>* csr_row, VectorBase<int>* csr_col, size_t& nNNZ);

template void sparseMatrixTranspose(VectorBase<double>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<double>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows);
template void sparseMatrixTranspose(VectorBase<float>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<float>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows);
template void sparseMatrixTranspose(VectorBase<int>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<int>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows);
template void sparseMatrixTranspose(VectorBase<size_t>* resval, VectorBase<int>* resrow, VectorBase<int>* rescol, VectorBase<size_t>* inval, VectorBase<int>* inrow, VectorBase<int>* incol, size_t res_NNZ, size_t in_NNZ, size_t in_num_rows);

template void sparseMatrixVectorMultiply(float* Aval, int* Arow, int* Acol, float* x, float* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);
template void sparseMatrixVectorMultiply(double* Aval, int* Arow, int* Acol, double* x, double* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);
template void sparseMatrixVectorMultiply(int* Aval, int* Arow, int* Acol, int* x, int* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);
template void sparseMatrixVectorMultiply(size_t* Aval, int* Arow, int* Acol, size_t* x, size_t* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);

template void setDiagonal(double* a, const double number, size_t offset, size_t M, size_t N);
template void setDiagonal(float* a, const float number, size_t offset, size_t M, size_t N);
template void setDiagonal(int* a, const int number, size_t offset, size_t M, size_t N);
//template void setDiagonal(size_t* a, const int number, size_t offset, size_t M, size_t N);

template void getDiagonal(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getDiagonal(float* a, const float* b, size_t offset, size_t M, size_t N);
template void getDiagonal(int* a, const int* b, size_t offset, size_t M, size_t N);

template void getLowerTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getLowerTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);
template void getLowerTriangular(int* a, const int* b, size_t offset, size_t M, size_t N);

template void getUpperTriangular(double* a, const double* b, size_t offset, size_t M, size_t N);
template void getUpperTriangular(float* a, const float* b, size_t offset, size_t M, size_t N);
template void getUpperTriangular(int* a, const int* b, size_t offset, size_t M, size_t N);


template void sparseMatrixMatrixMultiply(double* Aval, int* Arow, int* Acol, double* Bval, int* Brow, int* Bcol,
    double* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
    size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);
template void sparseMatrixMatrixMultiply(float* Aval, int* Arow, int* Acol, float* Bval, int* Brow, int* Bcol,
    float* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
    size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);
template void sparseMatrixMatrixMultiply(int* Aval, int* Arow, int* Acol, int* Bval, int* Brow, int* Bcol,
    int* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
    size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);
template void sparseMatrixMatrixMultiply(size_t* Aval, int* Arow, int* Acol, size_t* Bval, int* Brow, int* Bcol,
    size_t* Cval, int* Crow, int* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
    size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);

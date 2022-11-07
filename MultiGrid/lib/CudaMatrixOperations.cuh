#ifndef CUDAMATRIXOPERATIONS_H
#define CUDAMATRIXOPERATIONS_H

// These assume a column major layout of the matrix
void matrixMultiply(const float* a, const float* b, float* c, size_t N, size_t M, size_t K);
void matrixMultiply(const double* a, const double* b, double* c, size_t N, size_t M, size_t K);
void matrixMultiply(const int* a, const int* b, int* c, size_t N, size_t M, size_t K);

// TODO: Make cusparse functions also work with double.
void sparseMatrixVectorMultiply(void* Aval, void* Arow, void* Acol, void* x, void* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ);

// The offset is the offset from main diagonal, if it's positive then it's an offset along the rows and if negative
// then an offset along the column.
void sparseMatrixMatrixMultiply(void* Aval, void* Arow, void* Acol, void* Bval, void* Brow, void* Bcol,
	void* Cval, void* Crow, void* Ccol, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ,
	size_t B_num_rows, size_t B_num_cols, size_t B_NNZ, size_t& C_num_rows, size_t& C_num_cols, size_t& C_NNZ);

// Converts dense matrix to CSR format
void convertDensetoCSR(void* dense, size_t nrows, size_t ncols, void* csr_val, void* csr_row, void* csr_col, size_t& nNNZ);

// Sets a chosen diagonal of a matrix with diagonal chosen by the offset.  Lower diagonal can be selected by choosing offset = (1,..,row-1)
// Upper diagonal can be chosen by choosing offset = (1,...,row-1) * row.
template <class T> void setDiagonal(T* a, const T number, size_t offset, size_t M, size_t N);

// Get diagonal of b and sets it to diagonal of a.
template <class T> void getDiagonal(T* a, const T* b, size_t offset, size_t M, size_t N);
//Get lower trinagular form of b and set it to a.  Offset of 0 means that the diagonal is also 0'd out.
template <class T> void getLowerTriangular(T* a, const T* b, size_t offset, size_t M, size_t N);
//Get upper trinagular form of b and set it to a.  Offset of 0 means that the diagonal is also 0'd out.
template <class T> void getUpperTriangular(T* a, const T* b, size_t offset, size_t M, size_t N);

#endif
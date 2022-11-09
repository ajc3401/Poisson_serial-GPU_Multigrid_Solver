// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef CRS_MATRIX_H
#define CRS_MATRIX_H

//#include "Vector.h"
#include "VectorBase.h"
#include "Matrix.h"

// Class to hold sparse matrices in CRS format.
template <class T>
class CRS_Matrix
{
public:
	CRS_Matrix(size_t _nNNZ, size_t _nrow, size_t _ncol);
	CRS_Matrix(const CRS_Matrix<T>& copy);
	CRS_Matrix(CRS_Matrix<T>&& move) noexcept;
	~CRS_Matrix();

	CRS_Matrix& operator=(const CRS_Matrix<T>& rhs);
	CRS_Matrix& operator=(CRS_Matrix<T>&& rhs) noexcept;
	CRS_Matrix& operator-();

	// Special method used to set up matrix discretizing poisson equation, but not necessarily just for that purpose
	void set_tridiagonal(const T bottomdiag_number, const T diag_number, const T upperdiag_number);
	// Methods to set up diagonal, upper diagonal, and lower diagonal.
	void set_diagonal(const T diag_number);
	void set_lowerdiagonal(const T bottomdiag_number);
	void set_upperdiagonal(const T upperdiag_number);

	void sparse_mat_mul(const CRS_Matrix<T>& A, const CRS_Matrix<T>& B, bool allocate_more_space);
	void convert_to_crs(const Matrix<T>& dense_mat);
	void tranpose(const CRS_Matrix<T>& input);
	void invert_elements();
	// Obtain critical parts of matrix.
	/*void get_diagonal(const CRS_Matrix<T>& inmat);
	void get_lower_triangular(const CRS_Matrix<T>& inmat);
	void get_upper_triangular(const CRS_Matrix<T>& inmat);*/

	//void mat_vec_mul(const CRS_Matrix<T>& a, const Vector<T>& b);

	void swap(CRS_Matrix<T>& a, CRS_Matrix<T>& b);

	void display_valptr() const;
	void display_rowptr() const;
	void display_colptr() const;

	inline size_t get_ncol() const { return this->m_ncol; }
	inline size_t get_nrow() const { return this->m_nrow; }
	inline size_t get_nNNZ() const { return this->m_nNNZ; }
	inline VectorBase<T>* get_valptr() const { return this->m_crsValPtr; }
	inline VectorBase<int>* get_rowptr() const { return this->m_crsRowPtr; }
	inline VectorBase<int>* get_colptr() const { return this->m_crsColPtr; }

private:
	size_t m_nNNZ;
	size_t m_nrow;
	size_t m_ncol;
	VectorBase<int>* m_crsRowPtr;
	VectorBase<int>* m_crsColPtr;
	VectorBase<T>* m_crsValPtr;
};

#endif
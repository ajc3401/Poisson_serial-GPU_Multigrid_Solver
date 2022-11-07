#ifndef MATRIX_H
#define MATRIX_H
#include "Grid.h"
#include "VectorBase.h"


template <class T>
class Matrix
{
public:
	Matrix(const size_t _nrow, const size_t _ncol, bool fill_with_zeros = true);
	// This constructor will be used to initialize the "A" matrix for different differential equations
	Matrix(const std::string& differential_equation, Grid _grid);
	Matrix(const Matrix<T>& copy);
	Matrix(Matrix<T>&& move) noexcept;

	~Matrix();

	// Matrix operations
	Matrix& operator=(const Matrix<T>& rhs);
	Matrix& operator=(Matrix<T>&& rhs) noexcept;

	Matrix& operator-=(const Matrix<T>& rhs);
	Matrix& operator+=(const Matrix<T>& rhs);

	void invert_elements();
	void mat_mul(const Matrix<T>& a, const Matrix<T>& b);
	
	// Setters for matrix elements
	void set_to_number(const T number);
	void set_diagonal(const T number, size_t offset);
	inline void set_ncol_nrow(const size_t _ncol, const size_t _nrow) { m_ncol = _ncol; m_nrow = _nrow; }
	

	// Utility functions
	void swap(Matrix<T>& a, Matrix<T>& b);
	void display() const;

	// Private member getters
	inline size_t get_ncol() const { return this->m_ncol; }
	inline size_t get_nrow() const { return this->m_nrow; }
	inline VectorBase<T>* get_vecptr() const { return this->m_vecptr; }

	// Extract different parts of a matrix and set them as the elements of a new matrix
	void get_diagonal(const Matrix<T>& inmat, size_t offset);
	void get_lower_triangular(const Matrix<T>& inmat, size_t offset);
	void get_upper_triangular(const Matrix<T>& inmat, size_t offset);

	//T& operator[](size_t index) { assert(index < m_grid.get_totalnpoints()); return m_vecptr->begin()[index]; };
	//T const& operator[](size_t index) const { assert(index < m_grid.get_totalnpoints()); return m_vecptr->begin()[index]; };
	
	
	// Friend functions to set the elements for the "A" matrix for different differential equations
	// They are named according to differential equation
	
private:
	size_t m_nrow;
	size_t m_ncol;
	VectorBase<T>* m_vecptr;
};

#endif
#include "Matrix.h"

#ifdef __GPU__
#include "CudaVector.cuh"
#include "CudaMatrixOperations.cuh"
#else
#include "SerialVector.h"
#include "SerialMatrixOperations.h"
#endif

template<class T>
Matrix<T>::Matrix(const size_t _nrow, const size_t _ncol, bool fill_with_zeros) :
	m_nrow{ _nrow },
	m_ncol{ _ncol },
	m_vecptr{nullptr}
{
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(m_nrow * m_ncol);
#else 
	m_vecptr = new SerialVector<T>(m_nrow * m_ncol);
#endif
	if(fill_with_zeros)
		m_vecptr->resize(m_nrow * m_ncol, m_nrow * m_ncol);
	
}

template<class T>
Matrix<T>::Matrix(const std::string& differential_equation, Grid _grid) :
	m_nrow{ _grid.get_totalnpoints() },
	m_ncol{ _grid.get_totalnpoints() },
	m_vecptr{ nullptr }
{
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(m_nrow * m_ncol);
#else 
	m_vecptr = new SerialVector<T>(m_nrow * m_ncol);
#endif
}

template<class T>
Matrix<T>::Matrix(const Matrix<T>& copy) :
	m_nrow{ copy.m_nrow },
	m_ncol{ copy.m_ncol },
	m_vecptr{ nullptr }
{
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(m_nrow * m_ncol);
#else
	m_vecptr = new SerialVector<T>(m_nrow * m_ncol);
#endif
	/*for (size_t i = 0; i < copy.m_vecptr->get_nelem(); i++)
	{
		m_vecptr[i] = copy.m_vecptr[i];
	}*/
	for (size_t i = 0; i < copy.m_vecptr->get_nelem(); i++)
	{
		m_vecptr->push_back(copy.m_vecptr->begin()[i]);
	}
}

template<class T>
Matrix<T>::Matrix(Matrix<T>&& move) noexcept
{
	swap(*this, move);
}

template<class T>
Matrix<T>::~Matrix()
{
	if (m_vecptr)
		delete m_vecptr;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& rhs)
{
	Matrix<T> tmp(rhs);
	swap(*this, tmp);

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& rhs) noexcept
{

	swap(*this, rhs);

	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs)
{
	assert(this->m_ncol == rhs.m_ncol && this->m_nrow == rhs.m_nrow);
#ifdef __GPU__
	CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(rhs.m_vecptr);
#else
	SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
	SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(rhs.m_vecptr);
#endif
	* tmp1 += *tmp2;
	return *this;
}

template<class T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs)
{
	assert(this->m_ncol == rhs.m_ncol && this->m_nrow == rhs.m_nrow);
#ifdef __GPU__
	CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(rhs.m_vecptr);
#else
	SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
	SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(rhs.m_vecptr);
#endif
	* tmp1 -= *tmp2;
	return *this;
}

template<class T>
void Matrix<T>::mat_mul(const Matrix<T>& a, const Matrix<T>& b)
{
	assert(a.m_ncol == b.m_nrow);
	//Matrix<T> c(this->m_nrow, rhs.m_ncol);
#ifdef __GPU__
	const CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(a.m_vecptr);
	const CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(b.m_vecptr);
	//CudaVector<T>* tmp3 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
#else
	const SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(a.m_vecptr);
	const SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(b.m_vecptr);
	//SerialVector<T>* tmp3 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
#endif
	
	//MatrixMultiply(test2, test2, test2, test, test, test);
	matrixMultiply(tmp1->begin(), tmp2->begin(), this->m_vecptr->begin(), a.m_ncol, a.m_nrow, b.m_ncol);
	/*for (size_t i = 0; i < this->m_vecptr->get_nelem(); i++)
	{
		
		this->m_vecptr->begin()[i] = tmp3->begin()[i];
		
	}*/
	//this->m_vecptr->swap(*(this->m_vecptr), *(tmp3));
	//*(c.m_vecptr->begin()) = *(tmp3->begin());
	
}

template<class T>
void Matrix<T>::invert_elements()
{
	this->m_vecptr->invert_elements();
}

template<class T>
void Matrix<T>::set_to_number(const T number)
{
	this->m_vecptr->set_to_number(number);
}

// Offset goes down column, if you want upper diagonal use #row
template<class T>
void Matrix<T>::set_diagonal(const T number, size_t offset)
{
//#ifdef __GPU__
//	CudaVector<T>* tmp = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
//#else
//	SerialVector<T>* tmp = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
//#endif
	
	setDiagonal(this->m_vecptr->begin(), number, offset, this->m_nrow, this->m_ncol);
}

template <class T>
void Matrix<T>::get_diagonal(const Matrix<T>& inmat, size_t offset)
{
	assert((this->m_nrow == inmat.m_nrow) && (this->m_ncol == inmat.m_ncol));

	getDiagonal(this->m_vecptr->begin(), inmat.m_vecptr->begin(), offset, this->m_nrow, this->m_ncol);
}

template <class T>
void Matrix<T>::get_lower_triangular(const Matrix<T>& inmat, size_t offset)
{
	assert((this->m_nrow == inmat.m_nrow) && (this->m_ncol == inmat.m_ncol));

	getLowerTriangular(this->m_vecptr->begin(), inmat.m_vecptr->begin(), offset, this->m_nrow, this->m_ncol);
}

template <class T>
void Matrix<T>::get_upper_triangular(const Matrix<T>& inmat, size_t offset)
{
	assert((this->m_nrow == inmat.m_nrow) && (this->m_ncol == inmat.m_ncol));
	//int t{ 4 };
	//getLowerTriangular(this->m_vecptr->begin(), inmat.m_vecptr->begin(), offset, this->m_nrow, this->m_ncol);
	getUpperTriangular(this->m_vecptr->begin(), inmat.m_vecptr->begin(), offset, this->m_nrow, this->m_ncol);
	
}

template<class T>
void Matrix<T>::swap(Matrix<T>& a, Matrix<T>& b)
{
	std::swap(a.m_nrow, b.m_nrow);
	std::swap(a.m_ncol, b.m_ncol);
	std::swap(a.m_vecptr, b.m_vecptr);

}

template<class T>
void Matrix<T>::display() const
{
	
		for (size_t i = 0; i < m_nrow; i++)
		{
			for (size_t k = 0; k < m_ncol; k++)
			{
				std::cout << m_vecptr->begin()[i + m_nrow * k] << " ";
			}
			std::cout << std::endl;
		}
		
	
	
}

template class Matrix<double>;
template class Matrix<float>;

#include "Vector.h"

#ifdef __GPU__
#include "CudaVector.cuh"
#include "CudaMatrixOperations.cuh"
#include <cusparse.h>
#include <cuda_runtime_api.h>
#else
#include "SerialVector.h"
#include "SerialMatrixOperations.h"
#endif

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}

template<class T>
Vector<T>::Vector(Grid _grid) :
	m_grid{_grid},
	m_vecptr{nullptr}
{
	
	size_t N = _grid.get_totalnpoints();
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(N);
#else 
	m_vecptr = new SerialVector<T>(N);
#endif
	m_vecptr->resize(N, N);
}

template<class T>
Vector<T>::Vector(Grid _grid, const T a, const T b, std::string function) : 
	m_grid{ _grid },
	m_vecptr{ nullptr }
{
	size_t N = _grid.get_totalnpoints();
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(N);
#else 
	m_vecptr = new SerialVector<T>(N);
#endif
	//m_vecptr->resize(N, N);

	T h = (b - a) / (N - 1);
	T val{ a };
	
	for (size_t i = 0; i < N; i++)
	{
		m_vecptr->push_back(val);
		val += h;
	}

	if (function == std::string("sin"))
		m_vecptr->sin();
	else
		std::runtime_error("Only sin is supported.");

}

template<class T>
Vector<T>::Vector(const Vector<T>& copy) :
	m_grid{copy.m_grid},
	m_vecptr{ nullptr }
{
	size_t N = m_grid.get_totalnpoints();
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(N);
#else 
	m_vecptr = new SerialVector<T>(N);
#endif
	
	
	for (size_t i = 0; i < copy.m_vecptr->get_nelem(); i++)
	{
		m_vecptr->push_back(copy.m_vecptr->begin()[i]);
	}
	
	//* m_vecptr = *copy.m_vecptr;
}

template<class T>
Vector<T>::Vector(Vector<T>&& move) noexcept
{
	swap(*this, move);
	
}

template<class T>
Vector<T>::Vector(std::initializer_list<T> init) :
	m_grid{1,std::vector<size_t> {init.size()} },
	m_vecptr{nullptr}
{
#ifdef __GPU__
	m_vecptr = new CudaVector<T>(init);
#else
	m_vecptr = new SerialVector<T>(init);
#endif
}

template<class T>
Vector<T>::~Vector()
{
	if (m_vecptr)
		delete m_vecptr;
	
}

template<class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& rhs)
{
	Vector<T> tmp(rhs);
	swap(*this, tmp);

	return *this;
}

template<class T>
Vector<T>& Vector<T>::operator=(Vector<T>&& rhs) noexcept
{
	
	swap(*this, rhs);

	return *this;
}

template <class T>
Vector<T>& Vector<T>::operator-()
{
	*(this->m_vecptr) = -(* (this->m_vecptr));
	return *this;
}

template<class T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& rhs)
{
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
Vector<T>& Vector<T>::operator-=(const Vector<T>& rhs)
{
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
Vector<T>& Vector<T>::operator*=(const T& rhs)
{

	*(this->get_vecptr()) *= rhs;
	return *this;
}

template<class T>
T Vector<T>::dot_product(const Vector<T>& rhs) const
{
	T c{ 0 };
#ifdef __GPU__
	CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(rhs.m_vecptr);
#else
	SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
	SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(rhs.m_vecptr);
#endif
	c = tmp1->dot_product(*tmp2);
	return c;

}

template<class T>
T Vector<T>::l2norm() const
{
	T result{ 0 };
	result = this->m_vecptr->l2norm();
	return result;
}

template<class T>
void Vector<T>::set_to_number(const T number)
{
//#ifdef __GPU__
//	CudaVector<T>* tmp = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
//#else
//	SerialVector<T>* tmp = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
//#endif
	this->m_vecptr->set_to_number(number);
	/*for (size_t i = 0; i < tmp->get_nelem(); i++)
		std::cout << tmp->begin()[i] << std::endl;*/
	//*(this->m_vecptr) = *(tmp);
	
}

template<class T>
void Vector<T>::set_to_range(const Vector<T>& invec, size_t left, size_t right)
{
#ifdef __GPU__
	//CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	const CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(invec.m_vecptr);
#else
	//SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
	const SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(invec.m_vecptr);
#endif
	this->m_vecptr->set_to_range(left, right, *tmp2);

	//this->m_vecptr->swap(*(this->m_vecptr), *(tmp1));
}

template<class T>
void Vector<T>::resize(const size_t _nelem, const size_t _nalloc)
{
	this->m_vecptr->resize(_nelem, _nalloc);
}

template<class T>
void Vector<T>::resize(Grid _grid)
{
	size_t nelem(_grid.get_totalnpoints());
	this->m_vecptr->resize(nelem, nelem);
}

template<class T>
void Vector<T>::mat_mul(const Matrix<T>& a, const Vector<T>& b)
{
	assert(a.get_ncol() == b.m_vecptr->get_nelem());
	//Matrix<T> c(this->m_nrow, rhs.m_ncol);
#ifdef __GPU__
	const CudaVector<T>* tmp1 = dynamic_cast<CudaVector<T>*>(a.get_vecptr());
	const CudaVector<T>* tmp2 = dynamic_cast<CudaVector<T>*>(b.m_vecptr);
	//CudaVector<T>* tmp3 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
#else
	const SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(a.get_vecptr());
	const SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(b.m_vecptr);
	//SerialVector<T>* tmp3 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
#endif

	//MatrixMultiply(test2, test2, test2, test, test, test);
	matrixMultiply(tmp1->begin(), tmp2->begin(), this->m_vecptr->begin(), a.get_ncol(), a.get_nrow(), b.get_grid().get_totalnpoints());
	//for (size_t i = 0; i < this->m_vecptr->get_nelem(); i++)
	//{

	//	this->m_vecptr->begin()[i] = tmp3->begin()[i];

	//}
	//this->m_vecptr->swap(*(this->m_vecptr), *(tmp3));

	//*(c.m_vecptr->begin()) = *(tmp3->begin());

}
// We make a copy of the coarse vector to pass to the cuda/serial vector
// method.  This assumes "this" is the coarse vector.
template<class T>
void Vector<T>::interpolate(const Vector<T>& invec)
{
#ifdef __GPU__
//	CudaVector<T>* v_coarse = new CudaVector<T>(this->m_vecptr);// dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	const CudaVector<T>* v_coarse = dynamic_cast<CudaVector<T>*>(invec.m_vecptr);
#else
	SerialVector<T>* v_coarse = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
#endif
	
	if(this->m_grid.get_dim() == 1)
		this->m_vecptr->interpolate_1D(*v_coarse);
	else
		throw std::invalid_argument("Only 1D supported");
}

template<class T>
void Vector<T>::interject(const Vector<T>& invec)
{
#ifdef __GPU__
	//	CudaVector<T>* v_coarse = new CudaVector<T>(this->m_vecptr);// dynamic_cast<CudaVector<T>*>(this->m_vecptr);
	const CudaVector<T>* v_fine = dynamic_cast<CudaVector<T>*>(invec.m_vecptr);
#else
	const SerialVector<T>* v_fine = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
#endif

	if (this->m_grid.get_dim() == 1)
		this->m_vecptr->interject_1D(*v_fine);
	else
		throw std::invalid_argument("Only 1D supported");
}

template <class T>
void Vector<T>::sparse_mat_mul(const CRS_Matrix<T>& a, const Vector<T>& x)
{
	assert(a.get_ncol() == x.m_vecptr->get_nelem());
	
	sparseMatrixVectorMultiply(a.get_valptr()->begin(), a.get_rowptr()->begin(), a.get_colptr()->begin(), x.m_vecptr->begin(), this->m_vecptr->begin(), a.get_nrow(), a.get_ncol(), a.get_nNNZ());

}



template<class T>
void Vector<T>::swap(Vector<T>& a, Vector<T>& b)
{
	std::swap(a.m_grid, b.m_grid);
	std::swap(a.m_vecptr, b.m_vecptr);
	
}

template<class T>
void Vector<T>::display() const
{
	for (size_t i = 0; i < m_vecptr->get_nelem(); i++)
	{
		std::cout << m_vecptr->begin()[i] << " ";
	}
	std::cout << std::endl;
}

template class Vector<double>;
template class Vector<float>;
template class Vector<int>;
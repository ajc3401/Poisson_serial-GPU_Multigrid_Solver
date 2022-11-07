#include "CRS_Matrix.h"

#ifdef __GPU__
#include "CudaVector.cuh"
#include "CudaMatrixOperations.cuh"
#include "cusparse.h"
#else
#include "SerialVector.h"
#include "SerialMatrixOperations.h"
#endif

#ifdef __GPU__
	#define CHECK_CUSPARSE(func)                                                   \
	{                                                                              \
		cusparseStatus_t status = (func);                                          \
		if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
			printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
			exit(EXIT_FAILURE);                                                   \
    }                                                                          \
}
#endif
// Here we just allocate enough memory for each pointer
template <class T>
CRS_Matrix<T>::CRS_Matrix(size_t _nNNZ, size_t _nrow, size_t _ncol) :
	m_nrow{ _nrow },
	m_ncol{ _ncol },
	m_nNNZ{ _nNNZ },
	m_crsColPtr{ nullptr },
	m_crsRowPtr{ nullptr },
	m_crsValPtr{ nullptr }

{
#ifdef __GPU__
	m_crsColPtr = new CudaVector<int>(m_nNNZ);
	m_crsRowPtr = new CudaVector<int>(m_nrow+1);
	m_crsValPtr = new CudaVector<T>(m_nNNZ);
#else 
	m_crsColPtr = new SerialVector<int>(m_nNNZ);
	m_crsRowPtr = new SerialVector<int>(m_nrow+1);
	m_crsValPtr = new SerialVector<T>(m_nNNZ);
#endif
	// The first element is always 0.  NOTE Cusparse starts indexing from 0 for both row and col ptrs.
	
	//m_vecptr->resize(m_nrow* m_ncol, m_nrow* m_ncol);
}

template<class T>
CRS_Matrix<T>::CRS_Matrix(const CRS_Matrix<T>& copy) :
	m_nrow{ copy.m_nrow },
	m_ncol{ copy.m_ncol },
	m_nNNZ{ copy.m_nNNZ },
	m_crsColPtr{ nullptr },
	m_crsRowPtr{ nullptr },
	m_crsValPtr{ nullptr }
{
#ifdef __GPU__
	m_crsColPtr = new CudaVector<int>(m_ncol);
	m_crsRowPtr = new CudaVector<int>(m_nrow);
	m_crsValPtr = new CudaVector<T>(m_nNNZ);
#else 
	m_crsColPtr = new SerialVector<int>(m_ncol);
	m_crsRowPtr = new SerialVector<int>(m_nrow);
	m_crsValPtr = new SerialVector<T>(m_nNNZ);
#endif
	std::cout << "copy m_nNNZ = " << copy.m_nNNZ << std::endl;
	for (size_t i = 0; i < copy.m_crsColPtr->get_nelem(); i++)
	{
		m_crsColPtr->push_back(copy.m_crsColPtr->begin()[i]);
	}
	for (size_t i = 0; i < copy.m_crsRowPtr->get_nelem(); i++)
	{
		m_crsRowPtr->push_back(copy.m_crsRowPtr->begin()[i]);
	}
	for (size_t i = 0; i < copy.m_crsValPtr->get_nelem(); i++)
	{
		m_crsValPtr->push_back(copy.m_crsValPtr->begin()[i]);
	}
	std::cout << "Num of elems is " << this->m_crsRowPtr->get_nelem() << std::endl;
	


}

template<class T>
CRS_Matrix<T>::CRS_Matrix(CRS_Matrix<T>&& move) noexcept
{
	std::cout << "Move" << std::endl;
	swap(*this, move);
}

template<class T>
CRS_Matrix<T>::~CRS_Matrix()
{
	if (m_crsRowPtr)
		delete m_crsRowPtr;
	if (m_crsColPtr)
		delete m_crsColPtr;
	if (m_crsValPtr)
		delete m_crsValPtr;
}

template<class T>
void CRS_Matrix<T>::swap(CRS_Matrix<T>& a, CRS_Matrix<T>& b)
{
	std::swap(a.m_nrow, b.m_nrow);
	std::swap(a.m_ncol, b.m_ncol);
	std::swap(a.m_nNNZ, b.m_nNNZ);
	std::swap(a.m_crsColPtr, b.m_crsColPtr);
	std::swap(a.m_crsRowPtr, b.m_crsRowPtr);
	std::swap(a.m_crsValPtr, b.m_crsValPtr);

}

template<class T>
CRS_Matrix<T>& CRS_Matrix<T>::operator=(const CRS_Matrix<T>& rhs)
{
	CRS_Matrix<T> tmp(rhs);
	swap(*this, tmp);

	return *this;
}

template<class T>
CRS_Matrix<T>& CRS_Matrix<T>::operator=(CRS_Matrix<T>&& rhs) noexcept
{

	swap(*this, rhs);

	return *this;
}

template <class T>
CRS_Matrix<T>& CRS_Matrix<T>::operator-()
{
	*(this->m_crsValPtr) = -(*(this->m_crsValPtr));
	return *this;
}
// As opposed to setting it through calls to matrix operations, we can do this by pushing back elements to all
// the pointers
// Would prefer to use emplace back here but I can't figure out a way to make it virtual and overridable.
template<class T>
void CRS_Matrix<T>::set_tridiagonal(const T bottomdiag_number, const T diag_number, const T upperdiag_number)
{
	assert(m_ncol == m_nrow); // Currently this method only supports square matrices
	// Only these two are in the 1st row
	m_crsRowPtr->push_back(0);
	m_crsValPtr->push_back(diag_number);
	m_crsValPtr->push_back(upperdiag_number);
	// First two numbers are in columns 0 and 1
	m_crsColPtr->push_back(0);
	m_crsColPtr->push_back(1);

	m_crsRowPtr->push_back(2);
	
	// After 1st row, each successive row until the last one contains all three numbers
	// Row ptr goes [0, 2, 5, 8, ...]
	// Col ptr goes [0, 1, 0, 1, 2, 1, 2, 3 ...]
	for (size_t i = 1; i < m_nrow-1; i++)
	{
		m_crsValPtr->push_back(bottomdiag_number);
		m_crsValPtr->push_back(diag_number);
		m_crsValPtr->push_back(upperdiag_number);

		m_crsRowPtr->push_back(2 + 3 * i);
		
		std::cout << std::endl;
		m_crsColPtr->push_back(0 + (i - 1));
		m_crsColPtr->push_back(1 + (i - 1));
		m_crsColPtr->push_back(2 + (i - 1));
	}
	
	
	// The final row
	m_crsValPtr->push_back(bottomdiag_number);
	m_crsValPtr->push_back(diag_number);
	// Last row has one less than the previous pattern
	m_crsRowPtr->push_back(3 * m_nrow - 2);
	m_crsColPtr->push_back(m_ncol-2);
	m_crsColPtr->push_back(m_ncol-1);

	
}

template<class T>
void CRS_Matrix<T>::set_lowerdiagonal(const T bottomdiag_number)
{
	assert(m_ncol == m_nrow); // Currently this method only supports square matrices

	m_crsRowPtr->push_back(0);
	m_crsRowPtr->push_back(0);
	

	for (size_t i = 0; i < m_nrow-1; i++)
	{
		m_crsValPtr->push_back(bottomdiag_number);
		m_crsColPtr->push_back(i);
		m_crsRowPtr->push_back(i+1);
	}
}

template<class T>
void CRS_Matrix<T>::set_upperdiagonal(const T upperdiag_number)
{
	assert(m_ncol == m_nrow); // Currently this method only supports square matrices

	m_crsRowPtr->push_back(0);


	for (size_t i = 0; i < m_nrow - 1; i++)
	{
		m_crsValPtr->push_back(upperdiag_number);
		m_crsColPtr->push_back(i+1);
		m_crsRowPtr->push_back(i+1);
	}
}

template<class T>
void CRS_Matrix<T>::set_diagonal(const T diag_number)
{
	assert(m_ncol == m_nrow); // Currently this method only supports square matrices

	m_crsRowPtr->push_back(0);


	for (size_t i = 0; i < m_nrow; i++)
	{
		m_crsValPtr->push_back(diag_number);
		m_crsColPtr->push_back(i);
		m_crsRowPtr->push_back(i + 1);
	}
}

template<class T>
void CRS_Matrix<T>::convert_to_crs(const Matrix<T>& dense_mat)
{
	

	this->m_nrow = dense_mat.get_nrow();
	this->m_ncol = dense_mat.get_ncol();
	
#ifdef __GPU__
	
	
	
	
	convertDensetoCSR(dense_mat.get_vecptr()->begin(), this->m_nrow, this->m_ncol, this->m_crsValPtr->begin(), this->m_crsRowPtr->begin(), this->m_crsColPtr->begin(), this->m_nNNZ);
	// The cusparse method fills the pointers with values but we need to manually update the number of elements.
	this->m_crsValPtr->set_number_of_elements(this->m_nNNZ);
	this->m_crsColPtr->set_number_of_elements(this->m_nNNZ);
	this->m_crsRowPtr->set_number_of_elements(this->m_nrow + 1);
	
#else
	convertDensetoCSR(dense_mat.get_vecptr()->begin(), this->m_nrow, this->m_ncol, this->m_crsValPtr, this->m_crsRowPtr, this->m_crsColPtr, this->m_nNNZ);

#endif
}

template<class T>
void CRS_Matrix<T>::sparse_mat_mul(const CRS_Matrix<T>& A, const CRS_Matrix<T>& B, bool allocate_more_space)
{
	// If we need to allocate more space then we allot the max number of possible elements.
	if (allocate_more_space)
	{
		this->m_crsValPtr->reserve(A.get_nrow() * A.get_ncol());
		this->m_crsColPtr->reserve(A.get_nrow() * A.get_ncol());
	}
	//sparseMatrixTranspose
	sparseMatrixMatrixMultiply(A.get_valptr()->begin(), A.get_rowptr()->begin(), A.get_colptr()->begin(), B.get_valptr()->begin(), B.get_rowptr()->begin(), B.get_colptr()->begin(),
		this->get_valptr()->begin(), this->get_rowptr()->begin(), this->get_colptr()->begin(), A.get_nrow(), A.get_ncol(), A.get_nNNZ(),
		B.get_nrow(), B.get_ncol(), B.get_nNNZ(), this->m_nrow, this->m_ncol, this->m_nNNZ);
	
	this->m_crsValPtr->set_number_of_elements(this->m_nNNZ);
	this->m_crsColPtr->set_number_of_elements(this->m_nNNZ);
	this->m_crsRowPtr->set_number_of_elements(this->m_nrow + 1);
}

template <class T>
void CRS_Matrix<T>::tranpose(const CRS_Matrix<T>& input)
{
#ifdef __GPU__
	std::runtime_error("Tranpose operation is not available in the GPU implementation.");
#else
	std::cout << input.get_nNNZ() << std::endl;
	this->m_crsRowPtr->set_to_number(0);
	//this->m_crsColPtr->set_to_number(0);

	this->m_crsValPtr->set_to_number(0);
	sparseMatrixTranspose(this->get_valptr(), this->get_rowptr(), this->get_colptr(), input.get_valptr(), input.get_rowptr(),
		input.get_colptr(), this->m_nNNZ, input.get_nNNZ(), input.get_nrow());
#endif
	//this->m_crsValPtr->set_number_of_elements(input.get_nNNZ());
	//this->m_crsColPtr->set_number_of_elements(input.get_nNNZ());
	//this->m_crsRowPtr->set_number_of_elements(input.get_nrow() + 1);
}
//template <class T>
//void CRS_Matrix<T>::get_diagonal(const CRS_Matrix<T>& inmat)
//{
//	// The diagonal of a CRS matrix can be obtained by looking at the element of the value array that corresponds to 
//	// at most the nth instance of the column index on the nth row.  For example, on row 3 we expect a column index
//	// of 2 (index starts from 0) and so we look in the value array for at most the 3rd instance of a value that 
//	// corresponds to a column index of 2.  We get a max instance of 3 since there could be elements that 
//	// precede the diagonal value on the same row.  Thus for each row we must check how many instances of the column
//	// index there are.
//
//
//}
//template<class T>
//Vector<T>& mat_vec_mul(const Vector<T>& invec)
//{
//	assert(m_ncol == invec.get_grid().get_totalnpoints());
//	//Matrix<T> c(this->m_nrow, rhs.m_ncol);
//#ifdef __GPU__
//	CudaVector<void>* csrValPtr = dynamic_cast<CudaVector<T>*>(this->m_csrValPtr);
//	CudaVector<void>* csrRowPtr = dynamic_cast<CudaVector<size_t>*>(this->m_csrRowPtr);
//	CudaVector<void>* csrColPtr = dynamic_cast<CudaVector<size_t>*>(this->m_csrColPtr);
//	CudaVector<void>* xvecPtr = dynamic_cast<CudaVector<T>*>(invec.get_vecptr());
//	CudaVector<void>* yvecPtr = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
//	//CudaVector<T>* tmp3 = dynamic_cast<CudaVector<T>*>(this->m_vecptr);
//#else
//	SerialVector<void>* csrValPtr = dynamic_cast<SerialVector<T>*>(a.m_csrValPtr);
//	SerialVector<void>* csrRowPtr = dynamic_cast<SerialVector<size_t>*>(a.m_csrRowPtr);
//	SerialVector<void>* csrColPtr = dynamic_cast<SerialVector<size_t>*>(a.m_csrColPtr);
//	SerialVector<T>* vecPtr = dynamic_cast<SerialVector<T>*>(b.get_vecptr());
//	//const SerialVector<T>* tmp1 = dynamic_cast<SerialVector<T>*>(a.m_vecptr);
//	//const SerialVector<T>* tmp2 = dynamic_cast<SerialVector<T>*>(b.m_vecptr);
//	//SerialVector<T>* tmp3 = dynamic_cast<SerialVector<T>*>(this->m_vecptr);
//#endif
//	sparseMatrixVectorMultiply(csrValPtr, csrRowPtr, csrColPtr, void* x, void* y, size_t A_num_rows, size_t A_num_cols, size_t A_NNZ)
//}

template<class T>
void CRS_Matrix<T>::invert_elements()
{
	this->m_crsValPtr->invert_elements();
}

template<class T>
void CRS_Matrix<T>::display_valptr() const
{
	for (size_t i = 0; i < m_crsValPtr->get_nelem(); i++)
	{
		std::cout << m_crsValPtr->begin()[i] << " ";
	}
	std::cout << std::endl;
}

template<class T>
void CRS_Matrix<T>::display_rowptr() const
{
	for (size_t i = 0; i < m_crsRowPtr->get_nelem(); i++)
	{
		std::cout << m_crsRowPtr->begin()[i] << " ";
	}
	std::cout << std::endl;
}

template<class T>
void CRS_Matrix<T>::display_colptr() const
{
	for (size_t i = 0; i < m_crsColPtr->get_nelem(); i++)
	{
		std::cout << m_crsColPtr->begin()[i] << " ";
	}
	std::cout << std::endl;
}

//template class CRS_Matrix<int>;
template class CRS_Matrix<double>;
template class CRS_Matrix<float>;
template class CRS_Matrix<size_t>;
//template class CRS_Matrix<int>;
// As opposed to setting it through calls to matrix operations, we can do this by emplacing back elements to all
// the pointers
//template<class T>
//void CRS_Matrix<T>::set_diagonal(const T number, size_t offset)
//{
//	for (size_t i = 0; i < m_nNZZ; i++)
//	{
//		m_crsValPtr->emplace_back(number);
//		if(offset)
//		m_crsColPtr->emplace_back(i);
//	}
//}
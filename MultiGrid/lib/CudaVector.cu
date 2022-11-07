#include "CudaVector.cuh"
#include "CudaVectorOperations.cuh"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <cfloat>
#include "CudaMemoryHandler.cuh"

template <class T>
CudaVector<T>::CudaVector() : VectorBase<T>() 
{
	reserve(10);

}

template <class T>
CudaVector<T>::CudaVector(const size_t _nalloc) : VectorBase<T>	()
{
	reserve(_nalloc);
}

template<class T>
CudaVector<T>::CudaVector(std::initializer_list<T> init) : VectorBase<T>()
{
	
	resize(0, init.size());
	for (auto x : init)
	{
		this->push_back(x);
	}
	
}
//template <class T>
//CudaVector<T>::CudaVector(const SerialVector<T>& serial_vector) :
//	m_nalloc{ serial_vector.get_nalloc() },
//	m_nelem{ serial_vector.get_nelem() },
//	m_ptr{ CudaMemoryHandler<T>::allocate(serial_vector.get_nalloc()) }
//{
//	
//	for (size_t i = 0; i < serial_vector.get_nelem(); i++)
//	{
//		m_ptr[i] = serial_vector.begin()[i];
//	}
//}

template <class T>
CudaVector<T>::CudaVector(const CudaVector<T>& copy) : VectorBase<T> ()

{
	resize(copy.m_nelem, copy.m_nalloc);
	setEqual(this->m_ptr, copy.m_ptr, this->m_nelem);
	
}


template<class T>
CudaVector<T>::CudaVector(CudaVector<T>&& move) noexcept
{
	/*m_nalloc = move.m_nalloc;
	m_nelem = move.m_nelem;
	m_ptr = move.m_ptr;

	move.m_nalloc = 0;
	move.m_nelem = 0;
	move.m_ptr = nullptr;*/
	swap(*this, move);
}

template <class T>
CudaVector<T>::~CudaVector()
{
	for (size_t i = 0; i < m_nelem; i++)
	{
		m_ptr[m_nelem - 1 - i].~T();
	}
	CudaMemoryHandler<T>::deallocate(m_ptr);
}

template<class T>
CudaVector<T>& CudaVector<T>::operator=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	if (dynamic_cast<const CudaVector<T>*>(&rhs))
		*(this->m_ptr) = *(rhs.begin());
	else
		throw std::invalid_argument("Cannot equate a serial and GPU vector");
	
	return *this;
//	
}

template<class T>
CudaVector<T>& CudaVector<T>::operator=(const CudaVector<T>& rhs)
{
	CudaVector<T> tmp(rhs);
	swap(*this, tmp);
	return *this;
}

template<class T>
CudaVector<T>& CudaVector<T>::operator-()
{
	
	setNegative(this->m_ptr, this->m_nelem);
	return *this;
}
//template <class U>
//Vector< CudaType <U>> Vector< CudaType <U>>::operator+(const Vector< CudaType<U>> a, const Vector< CudaType<U>>& b)
//{
//	int N = rhs.m_nelem;
//	assert(m_nelem == N);
//	//U* tmp2 = static_cast<U*>(::operator new(N * sizeof(U)));
//	Vector< CudaType <U>> tmp(N);
//	VectorAddEqual<< <(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (this->m_ptr, rhs.m_ptr, tmp.m_ptr, N);
//	
//	cudaDeviceSynchronize();
//	
//	//for (size_t i = 0; i < N; i++)
//	//{
//	//	std::cout << tmp.m_ptr[i] << std::endl;
//	//}
//	//std::swap(this->m_ptr, tmp2);
//	swap(*this, tmp);
//	for (size_t i = 0; i < N; i++)
//	{
//		std::cout << this->m_ptr[i] << std::endl;
//	}
//	return *this;
//}

template <class T>
CudaVector<T>& CudaVector<T>::operator+=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&rhs);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot add a serial vector to a GPU vector");
	else
		(*this) += rhs;
	//VectorAdd << <(rhs.m_nelem - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (this->m_ptr, rhs.m_ptr, rhs.m_nelem);



	return *this;
}

template <class T>
CudaVector<T>& CudaVector<T>::operator-=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&rhs);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot add a serial vector to a GPU vector");
	else
		(*this) -= rhs;
	//VectorAdd << <(rhs.m_nelem - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >> > (this->m_ptr, rhs.m_ptr, rhs.m_nelem);



	return *this;
}



template <class T>
CudaVector<T>& CudaVector<T>::operator+=(const CudaVector<T>& rhs)
{
	assert(this->m_nelem == rhs.m_nelem);
	sumVectors(this->m_ptr, rhs.m_ptr, this->m_nelem);

	return *this;
}

template <class T>
CudaVector<T>& CudaVector<T>::operator-=(const CudaVector<T>& rhs)
{
	assert(this->m_nelem == rhs.m_nelem);
	subtractVectors(this->m_ptr, rhs.m_ptr, this->m_nelem);

	return *this;
}

template <class T>
CudaVector<T>& CudaVector<T>::operator*=(const T& rhs)
{
	scalarVectorMultiply(this->m_ptr, rhs, this->m_nelem);
	return *this;
}

template <class T>
T CudaVector<T>::dot_product(const VectorBase<T>& rhs) const
{
	assert(this->m_nelem == rhs.get_nelem());
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&rhs);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot compute the dot product of a serial and GPU vector.");
	else
	{
		T c = 0;
		c = this->dot_product(rhs);
		return c;
	}
}

template <class T>
T CudaVector<T>::dot_product(const CudaVector<T>& rhs) const
{
	assert(this->m_nelem == rhs.m_nelem);
	T c{ 0 };
	const CudaVector<T> tmp(*this);
	dotProduct(tmp.m_ptr, rhs.m_ptr, c, this->m_nelem);
	return c;
}

template<class T>
void CudaVector<T>::invert_elements()
{
	invertElements(this->m_ptr, this->m_nelem);
}

template<class T>
void CudaVector<T>::interpolate_1D(const VectorBase<T>& v_coarser)
{
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&v_coarser);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot interpolate a serial vector to GPU vector");
	else
		this->interpolate_1D(*CudaVectorptr);
}

template<class T>
void CudaVector<T>::interject_1D(const VectorBase<T>& v_finer)
{
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&v_finer);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot interject a serial vector to GPU vector");
	else
		this->interject_1D(*CudaVectorptr);
}

template<class T>
T CudaVector<T>::l2norm()
{
	T* result = new T(0.0f);
	//const CudaVector<T> tmp(*this);
	//float* result = new float(0);
	//const CudaVector<float> tmp(*this);
	l2Norm(result, this->m_ptr, this->m_nelem);
	return *result;
}

template<class T>
void CudaVector<T>::sin()
{
	std::string s("sin");
	applyFunction(this->m_ptr, this->m_nelem, s);
}

template<class T>
void CudaVector<T>::interpolate_1D(const CudaVector<T>& v_coarser)
{
	const size_t N_finer{ 2 * this->m_nelem + 1 };
	const size_t N_coarser{ v_coarser.m_nelem };
	
	if(this->m_nelem != N_finer)
		this->resize(N_finer, N_finer);

	interpolate1D(this->m_ptr, v_coarser.m_ptr, N_coarser);
	
	this->m_ptr[0] = 0.5 * v_coarser.m_ptr[0];
	this->m_ptr[N_finer - 1] = 0.5 * v_coarser.m_ptr[N_coarser - 1];
}

template<class T>
void CudaVector<T>::interject_1D(const CudaVector<T>& v_finer)
{
	const size_t N_finer{ v_finer.m_nelem };
	const size_t N_coarser{ static_cast<size_t>(0.5*(N_finer -1) + 1)};
	std::cout << "N_coarser = " << N_coarser << std::endl;
	if(this->m_nelem != N_coarser)
		this->resize(N_coarser, N_coarser);

	this->m_ptr[0] = v_finer.m_ptr[0];
	std::cout << v_finer.m_ptr[N_finer - 1] << std::endl;
	interject1D(this->m_ptr, v_finer.m_ptr, N_coarser);
	
	this->m_ptr[N_coarser - 1] = v_finer.m_ptr[N_finer - 1];
	
}

template<class T>
void CudaVector<T>::set_to_number(const T number)
{
	//CudaVector<T> tmp(*this);
	setValue(this->m_ptr, number, this->m_nelem);
	//for (size_t i = 0; i < tmp.m_nelem; i++)
	//	this->m_ptr[i] = tmp.m_ptr[i];
}

template<class T>
void CudaVector<T>::set_to_range(size_t left, size_t right, const VectorBase<T>& invec)
{
	const CudaVector<T>* CudaVectorptr = dynamic_cast<const CudaVector<T>*>(&invec);
	if (CudaVectorptr == nullptr)
		throw std::invalid_argument("Cannot set elements of a GPU vector to a serial vector.");
	else
		this->set_to_range(left, right, *CudaVectorptr);
}

template<class T>
void CudaVector<T>::set_to_range(size_t left, size_t right, const CudaVector<T>& invec)
{
	size_t rangelength = right - left;
	if (this->m_nelem != rangelength)
		this->resize(rangelength, rangelength);
	setRange(left, right, this->m_ptr, invec.m_ptr);
}

template<class T>
void CudaVector<T>::push_back(const T& element)
{
	if (m_nelem < m_nalloc) {

		m_ptr[m_nelem] = element;
		++m_nelem;
	}
	else
	{
		this->resize(m_nalloc * 1.5);
		new (m_ptr + m_nelem) T(element);
		++m_nelem;
	}

}


template<class T>
void CudaVector<T>::push_back(T&& element)
{
	if (m_nelem < m_nalloc) {

		m_ptr[m_nelem] = std::move(element);
		++m_nelem;
	}
	else
	{
		this->resize(m_nalloc * 1.5);
		new (m_ptr + m_nelem) T(std::move(element));
		++m_nelem;
	}

}

template<class T>
template<typename... Args> T& CudaVector<T>::emplace_back(Args&&... args)
{
	if (this->m_nelem < this->m_nalloc) {
		this->m_ptr[this->m_nelem] = T(std::forward<Args>(args)...);
		++this->m_nelem;
	}
	else
	{
		this->resize(this->m_nalloc * 1.5);
		this->m_ptr[this->m_nelem] = T(std::forward<Args>(args)...);
		++this->m_nelem;
	}
	return this->m_ptr[this->m_nelem];
}

template<class T>
void CudaVector<T>::reserve(const size_t _nalloc)
{
	//assert(_nalloc > m_nelem);
	
	CudaMemoryHandler<T>::deallocate(this->m_ptr);
	this->m_ptr = nullptr;
	this->m_nalloc = this->m_nelem = 0;

	this->m_nalloc = _nalloc;

	this->m_ptr = CudaMemoryHandler<T>::allocate(_nalloc);
	//CudaMemoryHandler<T>::allocate(_nalloc, this->m_ptr);
	
}

template<class T>
void CudaVector<T>::resize(const size_t _nelem, const size_t _nalloc)
{
	std::cout << "this elem = " << this->m_nelem << std::endl;
	std::cout << "nelem = " << _nelem << std::endl;
	std::cout << "nalloc = " << _nalloc << std::endl;
	// We only allow growing, not shrinking
	assert((_nelem >= m_nelem) && (_nalloc >= m_nelem));

	CudaVector<T> tmp(_nalloc);
	if (m_nelem > 0)
	{
		for (auto& x : *this)
			tmp.emplace_back(x);
	}
	for (size_t i = m_nelem; i < _nelem; i++)
		tmp.emplace_back(0);
	
	//tmp.m_nelem = _nelem;

	swap(tmp, *this);
	
}

template<class T>
void CudaVector<T>::resize(const size_t _nalloc)
{
	
	//assert(_nalloc > this->m_nelem);
	assert(_nalloc > this->m_nelem);

	CudaVector<T> tmp(_nalloc);
	if (m_nelem > 0)
	{
		for (auto& x : *this)
			tmp.emplace_back(x);
	}


	swap(tmp, *this);

}

template<class T>
void CudaVector<T>::pop_back()
{
	if (m_nelem > 0)
	{
		this->m_nelem--;
		this->m_ptr[this->m_nelem].~T();
	}

}

template<class T>
void CudaVector<T>::swap(VectorBase<T>& a, VectorBase<T>& b)
{
	CudaVector<T>* CudaVectorptra = dynamic_cast<CudaVector<T>*>(&a);
	CudaVector<T>* CudaVectorptrb = dynamic_cast<CudaVector<T>*>(&b);
	if (CudaVectorptra == nullptr || CudaVectorptrb == nullptr)
		throw std::invalid_argument("Cannot swap a serial and GPU vector");
	else
		swap(a, b);
}


template<class T>
void CudaVector<T>::swap(CudaVector<T>& a, CudaVector<T>& b)
{

	std::swap(a.m_nalloc, b.m_nalloc);
	std::swap(a.m_nelem, b.m_nelem);
	std::swap(a.m_ptr, b.m_ptr);
}

template<class T>
void CudaVector<T>::display() const
{
	for (size_t i = 0; i < m_nelem; i++)
	{
		std::cout << this->m_ptr[i] << std::endl;
	}
}


//template Vector< CudaType<double>>::Vector(const size_t _nalloc);

template class CudaVector<double>;
template class CudaVector<float>;
template class CudaVector<size_t>;
template class CudaVector<int>;


#include "SerialVector.h"
#include "SerialMatrixOperations.h"
#include <math.h>
#include <cmath>

template<class T>
SerialVector<T>::SerialVector() : VectorBase<T> ()
{
	
	reserve(10);
}

template<class T>
SerialVector<T>::SerialVector(const size_t _nalloc) : VectorBase<T> ()

{
	reserve(_nalloc);
}

template<class T>
SerialVector<T>::SerialVector(const SerialVector<T>& copy) : VectorBase<T> ()
	
{
	resize(copy.m_nelem, copy.m_nalloc);
	for (auto x : copy)
	{
		this->push_back(x);
	}
	/*for (size_t i=0; i<this->m_nelem; i++)
	{
		this->m_ptr[i] = copy.m_ptr[i];
	}*/
}
template<class T>
SerialVector<T>::SerialVector(std::initializer_list<T> init) : VectorBase<T> ()
{
	reserve(init.size());
	for (auto x : init)
	{
		this->push_back(x);
	}
}


template<class T>
SerialVector<T>::SerialVector(SerialVector<T>&& move) noexcept
{
	/*this->m_nalloc = move.m_nalloc;
	this->m_nelem = move.m_nelem;
	this->m_ptr = move.m_ptr;

	move.m_nalloc = 0;
	move.m_nelem = 0;
	move.m_ptr = nullptr;*/
	swap(*this, move);
}

template<class T>
void SerialVector<T>::swap(VectorBase<T>& a, VectorBase<T>& b)
{
	SerialVector<T>* SerialVectorptra = dynamic_cast<SerialVector<T>*>(&a);
	SerialVector<T>* SerialVectorptrb = dynamic_cast<SerialVector<T>*>(&b);
	if (SerialVectorptra == nullptr || SerialVectorptrb == nullptr)
		throw std::invalid_argument("Cannot swap a serial and GPU vector");
	else
		swap(a, b);
}

template<class T>
void SerialVector<T>::swap(SerialVector<T>& a, SerialVector<T>& b)
{
	
	std::swap(a.m_nalloc, b.m_nalloc);
	std::swap(a.m_nelem, b.m_nelem);
	std::swap(a.m_ptr, b.m_ptr);
}


template<class T>
SerialVector<T>::~SerialVector()
{
	//delete[] m_ptr;
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[this->m_nelem - 1 - i].~T();
	}
	//::operator delete(m_ptr, m_nalloc * sizeof(T));//delete[] m_ptr;
}


template<class T>
void SerialVector<T>::push_back(const T& element)
{
	if (this->m_nelem < this->m_nalloc) {
		
		this->m_ptr[this->m_nelem] = element;
		++this->m_nelem;
	}
	else
	{
		this->resize(this->m_nalloc * 1.5);
		new (this->m_ptr + this->m_nelem) T(element);
		++this->m_nelem;
	}
	
}

template<class T>
void SerialVector<T>::push_back(T&& element)
{
	if (this->m_nelem < this->m_nalloc) {

		this->m_ptr[this->m_nelem] = std::move(element);
		++this->m_nelem;
	}
	else
	{
		this->resize(this->m_nalloc * 1.5);
		new (this->m_ptr + this->m_nelem) T(std::move(element));
		++this->m_nelem;
	}

}


template<class T>
template<typename... Args> T& SerialVector<T>::emplace_back(Args&&... args)
{
	if (this->m_nelem < this->m_nalloc) {
		this->m_ptr[this->m_nelem] = T(std::forward<Args>(args)...);
		++this->m_nelem;
	}
	else
	{
		this->resize(this->m_nalloc * 1.5, this->m_nalloc * 1.5);
		this->m_ptr[this->m_nelem] = T(std::forward<Args>(args)...);
		++this->m_nelem;
	}
	return this->m_ptr[this->m_nelem];
}

template<class T>
void SerialVector<T>::reserve(const size_t _nalloc)
{
	assert(_nalloc > this->m_nelem);
	std::cout << "nalloc = " << _nalloc << std::endl;
	//T* p = static_cast<T*>(::operator new(49 * sizeof(T)));
	delete[] this->m_ptr;
	this->m_ptr = nullptr;
	//for (size_t i = 0; i < this->m_nelem; i++)
	//	p[i] = this->m_ptr[i];

	
	this->m_nalloc = this->m_nelem = 0;

	this->m_nalloc = _nalloc;
	std::cout << "m_nalloc = " << this->m_nalloc << std::endl;
	this->m_ptr = static_cast<T*>(::operator new(_nalloc * sizeof(T)));
	//delete p;
	//return *this;
}

template<class T>
void SerialVector<T>::resize(const size_t _nelem, const size_t _nalloc)
{
	//assert(_nalloc > this->m_nelem);
	std::cout << "this elem = " << this->m_nelem << std::endl;
	std::cout << "nelem = " << _nelem << std::endl;
	std::cout << "nalloc = " << _nalloc << std::endl;


	assert((_nelem > this->m_nelem) && (_nalloc > this->m_nelem));

	SerialVector<T> tmp(_nalloc);
	if (this->m_nelem > 0)
	{
		for (auto& x : *this)
			tmp.emplace_back(x);
	}
	for (size_t i = this->m_nelem; i < _nelem; i++)
		tmp.emplace_back(0);

	
	
	
	swap(tmp, *this);
	
}

template<class T>
void SerialVector<T>::resize(const size_t _nalloc)
{
	assert(_nalloc > this->m_nelem);

	SerialVector<T> tmp(_nalloc);
	if (this->m_nelem > 0)
	{
		for (auto& x : *this)
			tmp.emplace_back(x);
	}


	swap(tmp, *this);

}

template<class T>
void SerialVector<T>::pop_back()
{
	if (this->m_nelem > 0)
	{
		this->m_nelem--;
		this->m_ptr[this->m_nelem].~T();
	}
		
}


template<class T>
void SerialVector<T>::display() const
{
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		std::cout << this->m_ptr[i] << std::endl;
	}
}

template<class T>
SerialVector<T>& SerialVector<T>::operator=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	//SerialVector<T>* SerialVectorptr = dynamic_cast<SerialVector<T>*>(&rhs);
	if (dynamic_cast<const SerialVector<T>*>(&rhs))
		*(this->m_ptr) = *(rhs.begin());
	else
		throw std::invalid_argument("Cannot equate a serial and GPU vector");
		
	/*SerialVector<T> tmp(m_nalloc);
	tmp.resize(m_nelem);
	for (size_t i = 0; i < m_nelem; i++)
	{
		tmp.m_ptr[i] = this->m_ptr[i] + rhs.m_ptr[i];
	}
	swap(*this, tmp);*/
	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator-()
{
	
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		
		this->m_ptr[i] = this->m_ptr[i] * (1 - (std::is_unsigned<T>::value ? 2 : 0));
	}
	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator+=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&rhs);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot add a serial and GPU vector");
	else
		(*this) += rhs;
	/*SerialVector<T> tmp(m_nalloc);
	tmp.resize(m_nelem);
	for (size_t i = 0; i < m_nelem; i++)
	{
		tmp.m_ptr[i] = this->m_ptr[i] + rhs.m_ptr[i];
	}
	swap(*this, tmp);*/
	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator-=(const VectorBase<T>& rhs)
{
	assert(this->m_nelem == rhs.get_nelem());
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&rhs);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot add a serial and GPU vector");
	else
		(*this) -= rhs;
	/*SerialVector<T> tmp(m_nalloc);
	tmp.resize(m_nelem);
	for (size_t i = 0; i < m_nelem; i++)
	{
		tmp.m_ptr[i] = this->m_ptr[i] + rhs.m_ptr[i];
	}
	swap(*this, tmp);*/
	return *this;
}

template<class T>
T SerialVector<T>::dot_product(const VectorBase<T>& rhs) const
{
	assert(this->m_nelem == rhs.get_nelem());
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&rhs);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot compute the dot product of a serial and GPU vector.");
	else
	{
		T c{ 0 };
		c = this->dot_product(rhs);
		return c;
	}
}

template<class T>
SerialVector<T>& SerialVector<T>::operator=(const SerialVector<T>& rhs)
{
	SerialVector<T> tmp(rhs);
	swap(*this, tmp);

	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator+=(const SerialVector<T>& rhs)
{
	assert(this->m_nelem == rhs.m_nelem);
	/*SerialVector<T> tmp(m_nalloc);
	tmp.resize(m_nelem);*/
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[i] += rhs.m_ptr[i];
	}
	
	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator-=(const SerialVector<T>& rhs)
{
	assert(this->m_nelem == rhs.m_nelem);
	
	
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[i] -= rhs.m_ptr[i];
	}
	
	return *this;
}

template<class T>
SerialVector<T>& SerialVector<T>::operator*=(const T& rhs)
{
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[i] *= rhs;
	}
	return *this;
}

template<class T>
T SerialVector<T>::dot_product(const SerialVector<T>& rhs) const
{
	assert(this->m_nelem == rhs.m_nelem);
	T c{ 0 };
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		c+= this->m_ptr[i] * rhs.m_ptr[i];
	}
	return c;
}

template<class T>
void SerialVector<T>::invert_elements()
{
	
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		if (this->m_ptr[i] > 0)
			this->m_ptr[i] = 1/this->m_ptr[i];
	}

}

template<class T>
void SerialVector<T>::interpolate_1D(const VectorBase<T>& v_coarser)
{
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&v_coarser);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot compute the dot product of a serial and GPU vector.");
	else
		this->interpolate_1D(*SerialVectorptr);
}

template<class T>
void SerialVector<T>::interject_1D(const VectorBase<T>& v_finer)
{
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&v_finer);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot compute the dot product of a serial and GPU vector.");
	else
		this->interpolate_1D(*SerialVectorptr);
}

template<class T>
void SerialVector<T>::interpolate_1D(const SerialVector<T>& v_coarser)
{
	// n = 2 * (v_coarser.m_nelem + 1)
	// v_finer = zeros(n - 1)
	const size_t N_finer{ 2 * this->m_nelem + 1 };
	const size_t N_coarser{ v_coarser.m_nelem };

	this->resize(N_finer, N_finer);

	

	for (size_t i = 0; i < N_coarser; i++)
	{
		this->m_ptr[2 * i + 1] = v_coarser.m_ptr[i];
		this->m_ptr[2 * (i + 1)] = 0.5 * (v_coarser.m_ptr[i - 1] + v_coarser.m_ptr[i]);
	}

	this->m_ptr[0] = 0.5 * v_coarser.m_ptr[0];
	this->m_ptr[N_finer - 1] = 0.5 * v_coarser.m_ptr[N_coarser - 1];

	

}

template<class T>
void SerialVector<T>::interject_1D(const SerialVector<T>& v_finer)
{
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&v_finer);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot compute the dot product of a serial and GPU vector.");
	else
		this->interpolate_1D(*SerialVectorptr);
}

template<class T>
T SerialVector<T>::l2norm()
{
	
	T result{ 0 };
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		result += this->m_ptr[i] * this->m_ptr[i];
	}
	result = sqrt(result);
	return result;
}

template<class T>
void SerialVector<T>::sin()
{
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[i] = std::sin(this->m_ptr[i]);
	}
}

template<class T>
void SerialVector<T>::set_to_number(const T number)
{
	for (size_t i = 0; i < this->m_nelem; i++)
	{
		this->m_ptr[i] = number;
	}
}



template<class T>
void SerialVector<T>::set_to_range(size_t left, size_t right, const VectorBase<T>& invec)
{
	const SerialVector<T>* SerialVectorptr = dynamic_cast<const SerialVector<T>*>(&invec);
	if (SerialVectorptr == nullptr)
		throw std::invalid_argument("Cannot set elements of a GPU vector to a serial vector.");
	else
		this->set_to_range(left, right, *SerialVectorptr);
}

template<class T>
void SerialVector<T>::set_to_range(size_t left, size_t right, const SerialVector<T>& invec)
{
	size_t rangelength = right - left;
	if (this->m_nelem != rangelength)
		this->resize(rangelength, rangelength);
	std::copy(invec.m_ptr + left, invec.m_ptr + right, this->m_ptr);
}
//template<class T>
//void SerialVector<T>::set_diagonal(const T number, size_t offset)
//{
//	
//}
//template class SerialVector<Dummy>;
template class SerialVector<int>;
template class SerialVector<double>;
template class SerialVector<float>;
template class SerialVector<size_t>;
//template class SerialVector<int>;
// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef VECTORBASE_H
#define VECTORBASE_H
#include <assert.h>
template<class T>
class VectorBase
{
public:
	VectorBase() : m_nalloc{ 0 }, m_nelem{ 0 }, m_ptr{ nullptr } {};

	virtual ~VectorBase() {};

	inline size_t get_nalloc() const { return this->m_nalloc; }
	inline size_t get_nelem() const { return this->m_nelem; }
	

	virtual T* begin() { return m_ptr; }
	virtual T* end() { return (m_ptr + m_nelem); }
	virtual T* begin() const { return m_ptr; }
	virtual T* end() const { return (m_ptr + m_nelem); }

	//virtual void swap(VectorBase<T>& a, VectorBase<T>& b)=0;
	virtual void push_back(T const& element)=0;
	virtual void push_back(T&& element) = 0;
	//template<typename... Args> virtual T& emplace_back(Args&&... args) = 0;
	virtual void reserve(const size_t _nalloc)=0;
	virtual void resize(const size_t _nelem, const size_t _nalloc)=0;
	virtual void resize(const size_t _nalloc) = 0;
	virtual void swap(VectorBase<T>& a, VectorBase<T>& b) = 0;
	virtual void pop_back() = 0;
	//virtual void swap(CudaVector<T>& a, CudaVector<T>& b)=0;
	//template<typename... Args> virtual T& emplace_back(Args&&... args)=0;

	virtual VectorBase& operator=(const VectorBase<T>& rhs)=0;
	virtual VectorBase& operator-() = 0;
	/*virtual VectorBase& operator+(const VectorBase<T>& rhs)=0;
	virtual VectorBase& operator-(const VectorBase<T>& rhs)=0;*/
	virtual VectorBase& operator+=(const VectorBase<T>& rhs) = 0;
	virtual VectorBase& operator-=(const VectorBase<T>& rhs) = 0;
	virtual VectorBase& operator*=(const T& rhs) = 0;

	virtual T dot_product(const VectorBase<T>& rhs) const=0;
	virtual void invert_elements() = 0;
	virtual void interpolate_1D(const VectorBase<T>& v_coarser) = 0;
	virtual void interject_1D(const VectorBase<T>& v_finer) = 0;
	virtual void sin() = 0;
	virtual T l2norm() = 0;

	virtual void set_to_number(const T number) = 0;
	virtual void set_to_range(size_t left, size_t right, const VectorBase<T>& invec) = 0;
	inline virtual void set_number_of_elements(size_t number) = 0 ;
	//virtual void set_diagonal(const T number, size_t offset) = 0;

	virtual T& operator[](size_t index) { assert(index < m_nelem); return m_ptr[index]; };
	virtual T const& operator[](size_t index) const { assert(index < m_nelem); return m_ptr[index]; };
	//virtual T dot_product(const VectorBase<T>& rhs) const = 0;

	//virtual void display() const = 0;
protected:
	T* m_ptr;
	size_t m_nalloc;
	size_t m_nelem;
};

#endif
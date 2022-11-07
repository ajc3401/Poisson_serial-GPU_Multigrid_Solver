#ifndef SERIALVECTOR_H
#define SERIALVECTOR_H
#include<assert.h>
#include "VectorBase.h"
#include <iostream>

template <class T>
class SerialVector : public VectorBase<T>
{
public:
	SerialVector();
	SerialVector(const size_t _nalloc); 
	SerialVector(const SerialVector<T>& copy);
	SerialVector(std::initializer_list<T> init);
	SerialVector(SerialVector<T>&& move) noexcept;
	virtual ~SerialVector();
	
	//Memory handling
	virtual void push_back (T const& element) override;
	virtual void push_back(T&& element) override;
	template<typename... Args> T& emplace_back(Args&&... args);
	//virtual void swap(VectorBase<T>& a, VectorBase<T>& b) override;
	virtual void reserve(const size_t _nalloc) override;
	virtual void resize(const size_t _nelem, const size_t _nalloc) override;
	virtual void resize(const size_t _nalloc) override;
	virtual void swap(VectorBase<T>& a, VectorBase<T>& b);
	virtual void pop_back() override;

	void swap(SerialVector<T>& a, SerialVector<T>& b);

	
	void display() const;

	// Operators
	
	virtual SerialVector& operator=(const VectorBase<T>& rhs) override;
	virtual SerialVector& operator-() override;
	//virtual SerialVector& operator=(VectorBase<T>&& rhs) override;
	virtual SerialVector& operator+=(const VectorBase<T>& rhs) override;
	virtual SerialVector& operator-=(const VectorBase<T>& rhs) override;
	virtual SerialVector& operator*=(const T& rhs) override;
	virtual T dot_product(const VectorBase<T>& rhs) const override;
	virtual void invert_elements() override;
	virtual void interpolate_1D(const VectorBase<T>& v_coarser) override;
	virtual void interject_1D(const VectorBase<T>& v_finer) override;
	virtual T l2norm() override;
	virtual void sin() override;

	virtual void set_to_number(const T number) override;
	virtual void set_to_range(size_t left, size_t right, const VectorBase<T>& invec) override;
	virtual void set_to_range(size_t left, size_t right, const SerialVector<T>& invec);
	inline virtual void set_number_of_elements(size_t number) override { this->m_nelem = number; }
	//virtual void set_diagonal(const T number, size_t offset) override;

	SerialVector& operator=(const SerialVector<T>& rhs);
	SerialVector& operator+=(const SerialVector<T>& rhs);
	SerialVector& operator-=(const SerialVector<T>& rhs);
	T dot_product(const SerialVector<T>& rhs) const;
	void interpolate_1D(const SerialVector<T>& v_coarser);
	void interject_1D(const SerialVector<T>& v_finer);
	
	//T operator*(const SerialVector<T>& rhs); // Dot product

};


#endif



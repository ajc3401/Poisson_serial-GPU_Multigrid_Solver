// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef VECTOR_H
#define VECTOR_H
#include "VectorBase.h"
#include "Grid.h"
#include "Matrix.h"
#include "CRS_Matrix.h"
#include <span>


template<class T>
class Vector
{
public:
	Vector(Grid _grid);
	//Vector(size_t nalloc);
	Vector(Grid _grid, const T a, const T b, std::string function);
	Vector(const Vector<T>& copy);
	Vector(Vector<T>&& move) noexcept;
	Vector(std::initializer_list<T> init);
	~Vector();
	Vector& operator=(const Vector<T>& rhs);
	Vector& operator=(Vector<T>&& rhs) noexcept;
	Vector& operator-();
	
	Vector& operator-=(const Vector<T>& rhs);
	Vector& operator+=(const Vector<T>& rhs);
	Vector& operator*=(const T& rhs);

	T dot_product(const Vector<T>& rhs) const;
	T l2norm() const;
	void set_to_number(const T number);
	void set_to_range(const Vector<T>& invec, size_t left, size_t right);
	
	void mat_mul(const Matrix<T>& a, const Vector<T>& b);
	void sparse_mat_mul(const CRS_Matrix<T>& a, const Vector<T>& b);
	void interpolate(const Vector<T>& invec);
	void interject(const Vector<T>& invec);

	void swap(Vector<T>& a, Vector<T>& b);
	void display() const;
	void resize(const size_t _nelem, const size_t _nalloc);
	void resize(Grid _grid);


	inline Grid get_grid() const { return this->m_grid; }
	inline VectorBase<T>* get_vecptr() const { return this->m_vecptr; }

	T& operator[](size_t index) { assert(index < m_grid.get_totalnpoints()); return m_vecptr->begin()[index]; };
	T const& operator[](size_t index) const { assert(index < m_vecptr->get_nelem()); return m_vecptr->begin()[index]; };
	
	

	//inline size_t const get_vecnelem{ return m_vecptr->get_nelem(); }
	//inline size_t const get_vecnalloc{ return m_vecptr->get_nelem(); }
	//void resize(const size_t _xelem, const size_t _yelem, const size_t _zelem);

private:
	VectorBase<T>* m_vecptr;
	Grid m_grid;
};

//template<size_t left, size_t right, class T>
//constexpr auto slice(Vector<T>&& container)
//{
//	if constexpr (right > 0)
//	{
//		return std::span(container.get_vecptr()->begin(std::forward<T>(container)) + left, container.get_vecptr()->begin(std::forward<T>(container)) + right);
//	}
//	else
//	{
//		return std::span(container.get_vecptr()->begin(std::forward<T>(container)) + left, container.get_vecptr()->end(std::forward<T>(container)) + right);
//	}
//}
#endif
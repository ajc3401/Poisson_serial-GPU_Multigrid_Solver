#ifndef CUDAVECTOR_CUH
#define CUDAVECTOR_CUH

#include <iostream>
#include "SerialVector.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "CudaMemoryHandler.cuh"
#include "VectorBase.h"

//template <class U> extern CudaMemoryHandler<U> Cuda_Memory_Handler;



template<class T>
class CudaVector : public VectorBase<T>
{
public:
	CudaVector();
	CudaVector(const size_t _nalloc);
	
	//CudaVector(const SerialVector<T>& serial_vector);
	CudaVector(const CudaVector<T>& copy);
	CudaVector(CudaVector<T>&& move) noexcept;
	CudaVector(std::initializer_list<T> init);
	virtual ~CudaVector();

	virtual CudaVector& operator=(const VectorBase<T>& rhs) override;
	virtual CudaVector& operator-() override;
	virtual CudaVector& operator+=(const VectorBase<T>& rhs) override;
	virtual CudaVector& operator-=(const VectorBase<T>& rhs) override;
	virtual CudaVector& operator*=(const T& rhs) override;

	virtual T dot_product(const VectorBase<T>& rhs) const override;
	virtual void invert_elements() override;
	virtual void interpolate_1D(const VectorBase<T>& v_coarser) override;
	virtual void interject_1D(const VectorBase<T>& v_finer) override;
	virtual T l2norm() override;
	virtual void sin() override;

	virtual void set_to_number(const T number) override;
	virtual void set_to_range(size_t left, size_t right, const VectorBase<T>& invec) override;
	virtual void set_to_range(size_t left, size_t right, const CudaVector<T>& invec);
	inline virtual void set_number_of_elements(size_t number) override { this->m_nelem = number; };
	//virtual T dot_product(const VectorBase<T>& rhs) const override;
	

	CudaVector& operator=(const CudaVector<T>& rhs);
	
	CudaVector& operator-=(const CudaVector<T>& rhs);
	CudaVector& operator+=(const CudaVector<T>& rhs);
	T dot_product(const CudaVector<T>& rhs) const;
	void interpolate_1D(const CudaVector<T>& v_coarser);
	void interject_1D(const CudaVector<T>& v_finer);
	
	
	/*virtual T* begin() override { return m_ptr; }
	virtual T* end() override { return (m_ptr + m_nelem); }
	virtual T* begin() const override { return m_ptr; }
	virtual T* end() const override { return (m_ptr + m_nelem); }*/

	//virtual void swap(VectorBase<T>& a, VectorBase<T>& b) override;
	virtual void push_back(T const& element) override;
	virtual void push_back(T&& element) override;
	virtual void reserve(const size_t _nalloc) override;
	virtual void resize(const size_t _nelem, const size_t _nalloc) override;
	virtual void resize(const size_t _nalloc) override;
	virtual void swap(VectorBase<T>& a, VectorBase<T>& b);
	virtual void pop_back() override;
	//virtual void swap(VectorBase<T>& a, VectorBase<T>& b);
	template<typename... Args> T& emplace_back(Args&&... args);

	void swap(CudaVector<T>& a, CudaVector<T>& b);

	void display() const;
	//void resize(const size_t _nalloc);
	//void reserve(const size_t _nalloc);
	//void push_back(U const& element);
	//void swap(Vector<CudaType <U>>& a, Vector<CudaType <U>>& b);


};
#endif

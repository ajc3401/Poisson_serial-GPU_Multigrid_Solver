// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef PDE_BASE_H
#define PDE_BASE_H
#include "../lib/Vector.h"
#include "../lib/Matrix.h"
#include "../lib/CRS_Matrix.h"
#include "../lib/Grid.h"
#include <memory>

// Base class for any PDE to inherit from.  Members are sparse matrices where A is the matrix that defined the problem Av = f, L is the lower triangular part of A
// U is the upper triangular part of A, and Dinv is the inverse of the diagonal of A.  We instantiate these members seperately since they are needed in the smoothing
// algorithm and we wish to avoid obtaining them within the multigrid scheme.
template<class T>
class PDE_Base
{
public:
	PDE_Base(Grid _grid) : 
		m_A(_grid.get_totalnpoints() - 1, _grid.get_totalnpoints(), _grid.get_totalnpoints()),
		m_L(_grid.get_totalnpoints() - 1, _grid.get_totalnpoints(), _grid.get_totalnpoints()),  // NNZ will be total grid points - 1
		m_U(_grid.get_totalnpoints() - 1, _grid.get_totalnpoints(), _grid.get_totalnpoints()),
		m_Dinv(_grid.get_totalnpoints() - 1, _grid.get_totalnpoints(), _grid.get_totalnpoints())
	{	
	}
	virtual void Solve() = 0;
	virtual void Set_ALUDinv(CRS_Matrix<T>& A, CRS_Matrix<T>& L, CRS_Matrix<T>& U, CRS_Matrix<T>& Dinv, const Matrix<T>& A_dense) = 0;

	inline CRS_Matrix<T> get_L() const { return this->m_L; }
	inline CRS_Matrix<T> get_U() const { return this->m_U; }
	inline CRS_Matrix<T> get_Dinv() const { return this->m_Dinv; }

	// Allows us to declare pointers to L,U,Dinv in the Multigrid class methods
	// Alternatively, we could also declare PDE classes as friend classes of Multigrid
	inline CRS_Matrix<T>* get_Aptr() { return &m_A; }
	inline CRS_Matrix<T>* get_Lptr() { return &m_L; }
	inline CRS_Matrix<T>* get_Uptr() { return &m_U; }
	inline CRS_Matrix<T>* get_Dinvptr() { return &m_Dinv; }
protected:
	// Av = f

	//Matrix<T>* m_pA;
	 //The only data we need to perform the calculations.  We actually need A to calculate residual in multigrid.
	CRS_Matrix<T> m_L;
	CRS_Matrix<T> m_U;
	CRS_Matrix<T> m_Dinv;
	CRS_Matrix<T> m_A;
	//size_t m_npoints;
	//Vector<T> m_v;
	//Vector<T> m_f;
	
};

#endif
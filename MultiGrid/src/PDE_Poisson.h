#ifndef PDE_POISSON_H
#define PDE_POISSON_H
#include "PDE_Base.h"

// TODO: Make the constructor do all the work of setting up all the necessary parts

// The memory management strategy is to only store the memory of L, U, Dinv in CRS format.
// The easiest way to initialize them is to first create the A matrix which we store as a pointer.
// This A matrix is then initialized to discretize the Poisson equation and then L, U, and Dinv 
// are extracted from it.  We do this since A is never explicitly needed in any calculations. 
// Once that's done, the pointer to A is deleted and we call the Solve function 
template <class T>
class PDE_Poisson : public PDE_Base<T>
{
public:
	PDE_Poisson(Grid _grid);
	virtual void Solve() override;
	virtual void Set_ALUDinv(CRS_Matrix<T>& A, CRS_Matrix<T>& L, CRS_Matrix<T>& U, CRS_Matrix<T>& Dinv, const Matrix<T>& A_dense) override;
private:
	// Want to use all base class members without needing getters
	using PDE_Base<T>::m_L;
	using PDE_Base<T>::m_U;
	using PDE_Base<T>::m_Dinv;
	using PDE_Base<T>::m_A;
};

#endif
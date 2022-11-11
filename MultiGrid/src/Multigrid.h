// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef MULTIGRID_H
#define MULTIGRID_H
#include "../lib/Vector.h"
#include "../lib/Matrix.h"
#include "../lib/CRS_Matrix.h"
#include "../lib/Grid.h"
#include "PDE_Base.h"
#include "PDE_Poisson.h"
#include "Iterative_Methods.h"
// Class that handles the solving of the multigrid for a given PDE.
// 
// The strategy is to preallocate/precompute all the needed vectors/matrices at every level of coarsening so the
// main algorithm focuses just on the linear algebra operations of smoothing, interpolating, and restricting.
// We could also store interpolators/restrictors as matrices instead of using the kernels currently available.
// 
// We store the solution vector and right hand vector as std vectors of Vector pointers where the std vector is of length m_ncoarsen to respresent
// all the levels in the multigrid.
//
// Thought: Make all PDE classes friends of multigrid to access its private members
// This class has no data members since all relevant data will be initialized by
// the PDE children classes.
template <class T, class PDEType>
class Multigrid
{
public:
	Multigrid(T h, size_t _nrank, size_t dim, const Vector<T>& seed, size_t _ncoarsen, const size_t n_GS2, const size_t n_Jac, const size_t n_outer, const T LBC = 0.0f, const T RBC = 0.0f);
	~Multigrid();
	void V_Cycle(size_t coarsenIdx);
	void Solve();
	// Based on coarsen index we get n where n-1 is the amount of non boundary points in the grid (1D only so far)
	// Use these methods within a for loop over coarsenIdx.  Te
	void Set_Restrictor(size_t coarsenIdx);
	void Set_Interpolator(size_t coarsenIdx);
	void Set_ALUDinv(size_t coarsenIdx);
	void Set_VandF(size_t coarsenIdx, const Vector<T>& seed);
	;
private:
	
	std::vector<Vector<T>*> m_v;
	std::vector<Vector<T>*> m_f;
	// We store restriction matrices for each level.  The interpolators could be the tranpose but transposing CSR matrices
	// creates COO matrices and at the moment I don't want to deal with that so I set them up manually.
	std::vector<CRS_Matrix<T>*> m_restrictor;
	std::vector<CRS_Matrix<T>*> m_interpolator;
	std::vector<CRS_Matrix<T>*> m_A;
	std::vector<CRS_Matrix<T>*> m_L;
	std::vector<CRS_Matrix<T>*> m_U;
	std::vector<CRS_Matrix<T>*> m_Dinv;
	// If we want to try different smoothers then these shouldn't be hard coded but instead we can make a Smoother class which these are members of
	size_t m_nGS2;
	size_t m_nJac;
	size_t m_nouter;
	size_t m_ncoarsen;
	size_t m_ndim;
	size_t m_nrank;
	T m_h;
};

#endif // !MULTIGRID_H
// Copyright 2022, Anthony Cooper, All rights reserved

#include "Multigrid.h"
#include <math.h>


// We initialize a PDE model pointer which starts at the finest grid
template<class T, class PDEType>
Multigrid<T,PDEType>::Multigrid(T h, size_t _nrank, size_t dim, const Vector<T>& seed, size_t _ncoarsen, const size_t n_GS2, const size_t n_Jac, const size_t n_outer, const T LBC, const T RBC) :
	m_ncoarsen{ _ncoarsen },
	m_v{ std::vector<Vector<T>*>(_ncoarsen, nullptr) },
	m_f{ std::vector<Vector<T>*>(_ncoarsen, nullptr) },
	m_A{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) },
	m_L{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) },
	m_U{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) }, //CRS_Matrix<T>* (nullptr) },
	m_Dinv{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) },
	m_nGS2{ n_GS2 },
	m_nJac{n_Jac},
	m_nouter{n_outer},
	m_restrictor{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) },
	m_interpolator{ std::vector<CRS_Matrix<T>*>(_ncoarsen, nullptr) },
	m_nrank{_nrank},
	m_ndim{dim},
	m_h{h}
	


{
	// Builds the restrictor matrices at each grid level
	
	// AND

	// Initialize solution and RHS vector at each level starting with the finest grid.  We restrict the grid
	// at the end of each loop to get the next coarse grid.  We start with an initial guess for the finest (v_initial)
	// but the next levels are all initialized to 0.

	//Grid _grid(m_ndim, std::vector<size_t>{_nrank - 1} );
	
	

	for (size_t coarsenIdx = 0; coarsenIdx < _ncoarsen; coarsenIdx++)
	{

		
		Set_Restrictor(coarsenIdx);
		
		Set_Interpolator(coarsenIdx);
		
		Set_ALUDinv(coarsenIdx);

	
		Set_VandF(coarsenIdx, seed);
		
	}

	
}

template<class T, class PDEType>
Multigrid<T, PDEType>::~Multigrid()
{
	for (size_t coarsenIdx = 0; coarsenIdx < m_ncoarsen; coarsenIdx++)
	{
		if (m_v[coarsenIdx])
			delete m_v[coarsenIdx];
		if (m_f[coarsenIdx])
			delete m_f[coarsenIdx];
		if (m_A[coarsenIdx])
			delete m_A[coarsenIdx];
		if (m_L[coarsenIdx])
			delete m_L[coarsenIdx];
		if (m_U[coarsenIdx])
			delete m_U[coarsenIdx];
		if (m_Dinv[coarsenIdx])
			delete m_Dinv[coarsenIdx];
		if (m_restrictor[coarsenIdx])
			delete m_restrictor[coarsenIdx];
		if (m_interpolator[coarsenIdx])
			delete m_interpolator[coarsenIdx];
	}
}

template<class T, class PDEType>
void Multigrid<T, PDEType>::Set_ALUDinv(size_t coarsenIdx)
{
	size_t n = m_nrank / (pow(2, coarsenIdx));
	Grid grid(m_ndim, std::vector<size_t>{n - 1});
	
	PDEType* pPDE = new PDEType(grid);
	
	m_A[coarsenIdx] = new CRS_Matrix<T>(*(pPDE->get_Aptr()));
	
	m_L[coarsenIdx] = new CRS_Matrix<T>(*(pPDE->get_Lptr()));
	
	m_U[coarsenIdx] = new CRS_Matrix<T>(*(pPDE->get_Uptr()));
	m_Dinv[coarsenIdx] = new CRS_Matrix<T>(*(pPDE->get_Dinvptr()));
	
	 delete pPDE;

}
template<class T, class PDEType>
void Multigrid<T, PDEType>::Set_VandF(size_t coarsenIdx, const Vector<T>& seed)
{
	size_t n = m_nrank / (pow(2,coarsenIdx));
	Grid grid(m_ndim, std::vector<size_t>{n - 1});
	
	
	
	m_f[coarsenIdx] = new Vector<T>(grid);

	if (coarsenIdx == 0)
	{
		if (grid != seed.get_grid())
			std::runtime_error("Seed's grid is not equal to the finest grid.");
		m_v[coarsenIdx] = new Vector<T>(seed);
		/*m_v[coarsenIdx][0] = LBC;
		m_v[coarsenIdx][m_grid.get_totalnpoints() - 1] = RBC;*/
	}
	else
	{
		m_v[coarsenIdx] = new Vector<T>(grid);
	}
	// Add an option to choose this.
	m_f[coarsenIdx]->set_to_number(0.0f);
}

template<class T, class PDEType>
void Multigrid<T, PDEType>::Set_Restrictor(size_t coarsenIdx)
{
	size_t n = m_nrank / (pow(2, coarsenIdx));
	std::unique_ptr<Matrix<T>> pR(new Matrix<T>(n * 0.5 - 1, n - 1, false));
	size_t nrows = pR->get_nrow();
	size_t ncols = pR->get_ncol();
	
	for (size_t rowIdx = nrows; rowIdx > 0; rowIdx--)
	{
		pR->get_vecptr()->push_back(1.0f);
		for (size_t i = 0; i < nrows - 1; i++)
			pR->get_vecptr()->push_back(0.0f);

		pR->get_vecptr()->push_back(2.0f);
		for (size_t i = 0; i < nrows - 1; i++)
			pR->get_vecptr()->push_back(0.0f);

		pR->get_vecptr()->push_back(1.0f);
	}
	//std::cout << "Restrictor = " << std::endl;
	//pR->display();
	//std::cout << std::endl;
	m_restrictor[coarsenIdx] = new CRS_Matrix<T>(ncols * nrows, nrows, ncols);
	m_restrictor[coarsenIdx]->convert_to_crs(*pR);
}

template<class T, class PDEType>
void Multigrid<T, PDEType>::Set_Interpolator(size_t coarsenIdx)
{
	size_t n = m_nrank / (pow(2, coarsenIdx));
	std::unique_ptr<Matrix<T>> pR(new Matrix<T>(n - 1, n * 0.5 - 1, false));
	size_t nrows = pR->get_nrow();
	size_t ncols = pR->get_ncol();
	
	for (size_t colIdx = ncols; colIdx > 0; colIdx--)
	{
		pR->get_vecptr()->push_back(1.0f);
		pR->get_vecptr()->push_back(2.0f);
		pR->get_vecptr()->push_back(1.0f);
		for (size_t i = 0; i < nrows - 1; i++)
			pR->get_vecptr()->push_back(0.0f);

		
	}
	//std::cout << "Interpolator = " << std::endl;
	//pR->display();
	//std::cout << std::endl;
	m_interpolator[coarsenIdx] = new CRS_Matrix<T>(ncols * nrows, nrows, ncols);
	m_interpolator[coarsenIdx]->convert_to_crs(*pR);
}

template<class T, class PDEType>
void Multigrid<T, PDEType>::Solve()
{
	
	V_Cycle(0);
	std::cout << "Final finest v = " << std::endl;
	m_v[0]->display();
	
	T residual_l2norm(0.0f);
	Vector<T>* residual = new Vector<T>(m_v[0]->get_grid());
	residual->sparse_mat_mul(*m_A[0], *m_v[0]);
	*residual -= *m_f[0];
	residual_l2norm = residual->l2norm();
	std::cout << "The final l2 norm = is " << residual_l2norm << std::endl;
	
}

template<class T, class PDEType>
void Multigrid<T, PDEType>::V_Cycle(size_t coarsenIdx)
{
	
	m_h *= pow(2, coarsenIdx);
	T h = pow(m_h, 2);
	*m_f[coarsenIdx] *= h;

	std::cout << "h = " << m_h << std::endl;
	if (coarsenIdx == m_ncoarsen - 1)
	{
		GS2(*m_A[coarsenIdx], * m_Dinv[coarsenIdx], *m_L[coarsenIdx], *m_U[coarsenIdx], *m_v[coarsenIdx], *m_f[coarsenIdx], m_nGS2, m_nJac, m_nouter);
	}
		
	else
	{
		
		GS2(*m_A[coarsenIdx], *m_Dinv[coarsenIdx], *m_L[coarsenIdx], *m_U[coarsenIdx], *m_v[coarsenIdx], *m_f[coarsenIdx], m_nGS2, m_nJac, m_nouter);
		// Calculate new f based on residual
		Vector<T> res(m_v[coarsenIdx]->get_grid());
		res.sparse_mat_mul(*m_A[coarsenIdx], *m_v[coarsenIdx]);
		res -= *m_f[coarsenIdx];
		res = -res;
		
		m_f[coarsenIdx + 1]->sparse_mat_mul(*m_restrictor[coarsenIdx],res);
		m_v[coarsenIdx + 1]->set_to_number(0.0f);

		
		V_Cycle(coarsenIdx + 1);
		
		// Correct v at current level from interpolation of v at the coarser level
		Vector<T> tmp(m_v[coarsenIdx]->get_grid());
		
		tmp.sparse_mat_mul(*m_interpolator[coarsenIdx], *m_v[coarsenIdx+1]);
		*m_v[coarsenIdx] += tmp;

		// Post smoothing
		GS2(*m_A[coarsenIdx], *m_Dinv[coarsenIdx], *m_L[coarsenIdx], *m_U[coarsenIdx], *m_v[coarsenIdx], *m_f[coarsenIdx], m_nGS2, m_nJac, m_nouter);
	}
	
	

	
}

template class Multigrid<double, PDE_Poisson<double>>;
template class Multigrid<float, PDE_Poisson<float>>;
//template class Vector<int>;
#include "Iterative_Methods.h"

template <class T>
void SGS2(const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer)
{
	// Initialize residual vector r = (f - Uv)
	Vector<T> r(f.get_grid());
	// Initialize Jacobi Richardson solution vector
	Vector<T> g(r.get_grid());
	g.set_to_number(0.0f);
	// tmp will hold intermediate calculations
	Vector<T> tmp(f.get_grid());
	Vector<T> v_int(v.get_grid());
	tmp.set_to_number(0.0f);
	// Get final element index for v
	size_t last{ v.get_vecptr()->get_nelem() - 1};
	v[0] = 0.0f;
	v[last] = 0.0f;
	
	for (size_t iGS2 = 0; iGS2 < n_GS2; iGS2++)
	{
		// exchange interface elements of current solution
		for (size_t iouter = 0; iouter < n_outer; iouter++)
		{
			// Computes r = (f - Uv)
			
			tmp.sparse_mat_mul(U, v);
			tmp -= f;
			
			r = -tmp;
			
			// We now perform an inner Jacobi Richardson relaxation
			// Initial guess for g
			g.sparse_mat_mul(Dinv, r);

			// Computes g^(iJac+1) = Dinv * (r - Lg^(iJac))
			for (size_t iJac = 0; iJac < n_Jac; iJac++)
			{
				
				
				tmp.sparse_mat_mul(L, g);
				tmp -= r;
				tmp = -tmp;

				g.sparse_mat_mul(Dinv, tmp);
			}
			
			// Update solution vector v^(iGS2+1) = v^(IGS2) + g
			v_int = v;
			v += g;
			
			// Enforce BC
			v[0] = 0.0f;
			v[last] = 0.0f;
			// Compute new residual vector for backward sweep
			
			tmp.sparse_mat_mul(L, v_int);
			tmp -= f;
			r = -tmp;
			
			// We now perform an inner Jacobi Richardson relaxation
			// Initial guess for g
			for (size_t iJac = 0; iJac < n_Jac; iJac++)
			{
				
				tmp.sparse_mat_mul(U, g);
				tmp -= r;
				tmp = -tmp;
				g.sparse_mat_mul(Dinv, tmp);
			}
			
			// Update solution vector v^(iGS2+1) = v^(IGS2) + g
			v += g;
			// Enforce BC
			v[0] = 0.0f;
			v[last] = 0.0f;
		}
	}
}

template <class T>
void GS2(const CRS_Matrix<T>& A, const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer)
{
	Vector<T> r(f.get_grid());
	// Initialize Jacobi Richardson solution vector
	Vector<T> g(r.get_grid());
	g.set_to_number(0.0f);
	// tmp will hold intermediate calculations
	Vector<T> tmp(f.get_grid());
	
	CRS_Matrix<T> DinvL(Dinv.get_nNNZ(), Dinv.get_nrow(), Dinv.get_ncol());
	DinvL.sparse_mat_mul(Dinv, L, false);
	
	tmp.set_to_number(0.0f);
	// Get final element index for v
	size_t last{ v.get_vecptr()->get_nelem() - 1 };
	v[0] = 0.0f;
	v[last] = 0.0f;

	for (size_t iGS2 = 0; iGS2 < n_GS2; iGS2++)
	{
		// exchange interface elements of current solution
		for (size_t iouter = 0; iouter < n_outer; iouter++)
		{

			tmp.sparse_mat_mul(A, v);
			
			tmp -= f;
			tmp = -tmp;
			
			// We scale r by Dinv
			r.sparse_mat_mul(Dinv, tmp);

			// We now perform an inner Jacobi Richardson relaxation
			// Initial guess for g
			g = r;
			// Computes g^(iJac+1) = Dinv * (r - Lg^(iJac))
			for (size_t iJac = 0; iJac < n_Jac; iJac++)
			{
				tmp.sparse_mat_mul(DinvL, g);
				tmp += g;
				tmp -= r;
				tmp = -tmp;
				g += tmp;
			}
			v += g;
			v[0] = 0.0f;
			v[last] = 0.0f;
		}
	}
}
// Simple Jacobi Relaxation algorithm 
template <class T> void JacobiRelaxation(const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_iter)
{
	Vector<T> tmp(v.get_grid());
	Vector<T> tmp2(v.get_grid());
	size_t last{ v.get_vecptr()->get_nelem() - 1 };
	v[0] = 0.0f;
	v[last] = 0.0f;

	for (size_t it = 0; it < n_iter; it++)
	{
		tmp.sparse_mat_mul(U, v);
		tmp2.sparse_mat_mul(L, v);
		tmp += tmp2;
		tmp -= f;
		tmp = -tmp;
		v.sparse_mat_mul(Dinv, tmp);

		v[0] = 0.0f;
		v[last] = 0.0f;
	}
}

template void SGS2(const CRS_Matrix<float>& Dinv, const CRS_Matrix<float>& L, const CRS_Matrix<float>& U, Vector<float>& v, const Vector<float>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);
template void SGS2(const CRS_Matrix<double>& Dinv, const CRS_Matrix<double>& L, const CRS_Matrix<double>& U, Vector<double>& v, const Vector<double>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);

template void GS2(const CRS_Matrix<float>& A, const CRS_Matrix<float>& Dinv, const CRS_Matrix<float>& L, const CRS_Matrix<float>& U, Vector<float>& v, const Vector<float>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);
template void GS2(const CRS_Matrix<double>& A, const CRS_Matrix<double>& Dinv, const CRS_Matrix<double>& L, const CRS_Matrix<double>& U, Vector<double>& v, const Vector<double>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);

template void JacobiRelaxation(const CRS_Matrix<float>& Dinv, const CRS_Matrix<float>& L, const CRS_Matrix<float>& U, Vector<float>& v, const Vector<float>& f, const size_t n_iter);
template void JacobiRelaxation(const CRS_Matrix<double>& Dinv, const CRS_Matrix<double>& L, const CRS_Matrix<double>& U, Vector<double>& v, const Vector<double>& f, const size_t n_iter);
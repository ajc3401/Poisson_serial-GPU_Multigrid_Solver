// Copyright 2022, Anthony Cooper, All rights reserved

#ifndef ITERATIVE_METHODS_H
#define ITERATIVE_METHODS_H
#include "../lib/Vector.h"
#include "../lib/Matrix.h"
#include "../lib/CRS_Matrix.h"

// This solves Av = f using a two-stage Gauss Seidel relaxation.  This method uses only D^-1 (inv diag), U (upper triangular) and L (lower triangular) where
// A = (L + D + U)
// CRS formatted matrixes are used because the matrices are sparse for the Poisson equation.
template <class T> void GS2(const CRS_Matrix<T>& A, const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);
// Symmetric version of GS2 (currently does not work)
template <class T> void SGS2(const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_GS2, const size_t n_Jac, const size_t n_outer);
// Standard Jacobi Relaxation
template <class T> void JacobiRelaxation(const CRS_Matrix<T>& Dinv, const CRS_Matrix<T>& L, const CRS_Matrix<T>& U, Vector<T>& v, const Vector<T>& f, const size_t n_iter);
#endif

// Copyright 2022, Anthony Cooper, All rights reserved

#include <stdio.h>
#include <assert.h>
#include <numbers>
#include "../lib/Vector.h"
#include "../lib/Matrix.h"
#include "../lib/CRS_Matrix.h"
#include "Iterative_Methods.h"
#include "PDE_Poisson.h"
#include "Multigrid.h"



int main()
{

	/*Matrix<float> M(4, 4);
	M.set_diagonal(2.0f, 0);
	M.set_diagonal(-1.0f, 1);
	M.set_diagonal(-1.0f, M.get_nrow());
	M.display();

	CRS_Matrix<float> crs_M(10, 4, 4);
	crs_M.convert_to_crs(M);
	crs_M.display_colptr();
	crs_M.display_rowptr();
	crs_M.display_valptr();
	Vector<float> vec({ 1.0f, 2.0f, 0.0f, 3.0f });
	Vector<float> vec2({0.0f, 0.0f, 0.0f, 0.0f});
	vec2.sparse_mat_mul(crs_M, vec);
	vec2.display();*/

	size_t n_rank{ 16 };
	size_t n_GS2{ 3 };
	size_t n_Jac{ 2 };
	size_t n_outer{ 5 };
	size_t dim{ 1 };

	const float pi = 3.14159265358979323846;
	
	Grid grid(1, std::vector<size_t>{n_rank - 1});

	Vector<float> seed(grid, 0.0f, 2*pi, "sin");

	size_t n_coarsen{ 2 };
	float h{ 0.01 };

	Multigrid<float,PDE_Poisson<float>> multigrid(h, n_rank, dim, seed, n_coarsen, n_GS2, n_Jac, n_outer);
	multigrid.Solve();
	//Grid grid(1, std::vector<size_t>{8});
	//Vector<float> test({ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 });
	//test.display();
	//Vector<float> test2(grid);
	//test2.interject(test);
	//test2.display();
	
	//pR->get_vecptr()->push_back(1.0f);
	/*Matrix<float> mat1(4, 4);
	mat1.set_diagonal(2.0f, 0);
	Matrix<float> mat2(4, 4);
	mat2.set_diagonal(3.0f, 0);

	CRS_Matrix<float> csrmat1(16, 4, 4);
	CRS_Matrix<float> csrmat2(16, 4, 4);
	CRS_Matrix<float> csrmat3(16, 4, 4);
	
	csrmat1.convert_to_crs(mat1);
	csrmat2.convert_to_crs(mat2);
	csrmat3.sparse_mat_mul(csrmat1, csrmat2, false);

	csrmat3.display_valptr();
	csrmat3.display_rowptr();
	csrmat3.display_colptr();*/
	//CRS_Matrix<float>* L = new CRS_Matrix<float>(*poisson.get_Lptr());
	//float* vptr = static_cast<float*>(::operator new(v.get_grid().get_totalnpoints() * v2.get_grid().get_totalnpoints() * sizeof(float)));
	//vptr[0] = v;
	
	//Vector<float> v2({ 1,2,3,4,5,6,7,8,9 });
	/*std::cout << "Finer = ";
	v2.display();
	v.interject(v2);
	std::cout << "Coarser = ";
	v.display();*/
	
	//std::cout << pi << std::endl;

	
	//vec.display();
	/*Matrix<float> A(grid.get_totalnpoints(), grid.get_totalnpoints());
	Matrix<float> U(grid.get_totalnpoints(), grid.get_totalnpoints());

	A.set_diagonal(2.0f, 0);
	A.set_diagonal(-1.0f, 1);
	A.set_diagonal(-1.0f, 4);
	A.display();
	U.get_upper_triangular(A,0);
	U.display();*/
	//Grid, v_initial, n_GS2, n_Jac, n_outer, LBC = 0.0f, RBC = 0.0f
	//PDE_Poisson<float> Poisson(grid, 1.0f, 2, 2, 2);
	//Poisson.Solve();
	//size_t num_elem{ 15 };
	////// Two endpoints that are BC so we only operate on those not on boundary
	//size_t num_changed_elem{ 15 };
	//Grid grid(1, std::vector<size_t> {num_elem});
	//CRS_Matrix<float> Dinv(num_changed_elem, num_changed_elem, num_changed_elem);
	//CRS_Matrix<float> A(num_changed_elem, num_changed_elem, num_changed_elem);
	////CRS_Matrix<float> Atest(num_changed_elem, num_changed_elem, num_changed_elem);
	//CRS_Matrix<float> L(num_changed_elem-1, num_changed_elem, num_changed_elem);
	//CRS_Matrix<float> U(num_changed_elem-1, num_changed_elem, num_changed_elem);
	//Dinv.set_diagonal(0.5f);
	//L.set_lowerdiagonal(-1.0f);
	//U.set_upperdiagonal(-1.0f);
	//
	////
	//Vector<float> v(grid, 0.0f, 2 * pi, "sin");
	//Vector<float> v2(grid, 0.0f, 2 * pi, "sin");
	//Vector<float> f(grid);
	//Vector<float> tmp(grid);

	//Matrix<float> Amat(num_changed_elem, num_changed_elem);
	//Amat.set_diagonal(2.0f, 0);
	//Amat.set_diagonal(-1.0f, 1);
	//Amat.set_diagonal(-1.0f, Amat.get_nrow());
	//A.convert_to_crs(Amat);
	//tmp.mat_mul(Amat, v);
	//
	//tmp.sparse_mat_mul(A, v);
	//
	////v.set_to_number(1.0f);
	//f.set_to_number(0.0f);

	//size_t n_GS2{ 1 };
	//size_t n_Jac{ 1 };
	//size_t n_outer{ 10 };

	//size_t n_it{ 10 };
	////Vector<float> tmp(f.get_grid());
	////
	////GS2(A, Dinv, L, U, v, f, n_GS2, n_Jac, n_outer);
	////v.display();
	////JacobiRelaxation(Dinv, L, U, v2, f, n_it);
	//std::cout << "Results are : " << std::endl;
	//std::cout << "v = ";
	//v.display();
	//std::cout << std::endl;
	//std::cout << "v2 = ";
	//v2.display();
	//std::cout << std::endl;
//	Vector<double> vec1{ 2,4,6,8};
//	//Vector<double> vec3{ };
//	//vec3 = vec1.slice<1, 2>(vec1);
//	//std::cout << vec3 << std::endl;
//	Vector<double> vec2{3};
//	//double t{ 0 };
//	//t = vec1.dot_product(vec2);
//	vec2.set_to_range(vec1, 0, 2);
//	std::cout << "vec 2 = ";
//	vec2.display();
//	//vec2 = slice<1, 2>(vec1);
//	//vec2 += vec1;
//	//vec2.set_to_number(5.0);
//	Grid grid(1, std::vector<size_t> {3});
//	Vector<double> vec3(grid);
//	Matrix<double> mat1(2, 2);
//
//	mat1.set_to_number(1.0);
//	/*Matrix<double> mat2(2, 3);
//	mat2.get_lower_triangular(mat1,0);
//	mat2.display();*/
//	Matrix<double> mat2(mat1);
//	Matrix<double> mat3(2, 2);
//	//Matrix<double> mat4(2, 2);
//	////mat3.set_diagonal(1.0, -1);
//	mat3.mat_mul(mat1, mat2);
////mat4.get_diagonal(mat3,0);
//	////vec3 += vec1;
//	////vec3.set_to_number(5.0);
//	mat3.display();
	//mat4.display();
	//mat3 += mat1;
	//mat3.display();
	//Vector<double> vec1();
	//vec
	//std::cout << cpu1[0] << std::endl;
	// cpu3.get_nelem() << std::endl;
	

	//CudaVector<double> gpu1(cpu1);
	//CudaVector<double> gpu2(cpu2);
	//
	////Vector<CudaType <double>> d(e);
	/*CudaVector <double> gpu3(10);
	gpu3.resize(10, 20);
	std::cout << gpu3.get_nalloc() << std::endl;
	std::cout << gpu3.get_nelem() << std::endl;
	CudaVector <double> gpu1(gpu3);
	std::cout << gpu1.get_nalloc() << std::endl;
	std::cout << gpu1.get_nelem() << std::endl;*/
	//gpu1.swap(gpu1, gpu2);
	//gpu1 += gpu2;
	//dotProduct(gpu1.begin(), gpu2.begin(), tmp, );
	//tmp = gpu1.dot_product(gpu2);
	//cpu3 = cpu1 + cpu2;
	//
	//tmp = gpu1.dot_product(gpu2);
	//cpu3.display();
	
	//gpu1.display();
	//std::cout << gpu3.get_nalloc() << std::endl;
	//std::cout << gpu3.get_nelem() << std::endl;

	//

	
	return 0;
}
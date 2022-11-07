#include "PDE_Poisson.h"
#include "Iterative_Methods.h"

template <class T>
PDE_Poisson<T>::PDE_Poisson(Grid _grid) :
	PDE_Base<T>(_grid)
{
	// Make a pointer to A that we can delete after converting to CRS format
	//std::unique_ptr<Matrix<T>> pA = static_cast<std::unique_ptr<Matrix<T>>>(::operator new(_grid.get_totalnpoints() * sizeof(Matrix<T>)));
	std::unique_ptr<Matrix<T>> pA(new Matrix<T>(_grid.get_totalnpoints(), _grid.get_totalnpoints()));
	// TODO: Put this into a class method
	if (_grid.get_dim() == 1)
	{
		pA->set_ncol_nrow(_grid.get_totalnpoints(), _grid.get_totalnpoints());
		pA->set_diagonal(2.0f, 0);
		pA->set_diagonal(-1.0f, 1);
		pA->set_diagonal(-1.0f, pA->get_nrow());
	}
	//pA->display();
	//m_f.set_to_number(0.0f);
	//// TODO: With more complicated initializations, this should also be done by a method
	//// that takes in a string mapping to a certain initialization
	//m_v.set_to_number(v_initial);

	//// TODO: Make better way to initialize BC since we also need dimension dependence

	//m_v[0] = LBC;
	//m_v[_grid.get_totalnpoints()-1] = RBC;

	Set_ALUDinv(m_A, m_L, m_U, m_Dinv, *pA);

}

template <class T>
void PDE_Poisson<T>::Set_ALUDinv(CRS_Matrix<T>& A, CRS_Matrix<T>& L, CRS_Matrix<T>& U, CRS_Matrix<T>& Dinv, const Matrix<T>& A_dense)
{
	A.convert_to_crs(A_dense);
	// Split A into LUD
	// Make A L U and D shared pointers?
	std::cout << "A dense row = ";
	A_dense.display();
	std::cout << std::endl;
	/*Matrix<T> L_dense(A_dense.get_nrow(), A_dense.get_ncol());
	if (A_dense.get_nrow() == 7)
		std::cout << "A dense row = " << A_dense.get_nrow() << std::endl;
	Matrix<T> U_dense(A_dense.get_nrow(), A_dense.get_ncol());
	Matrix<T> D_dense(A_dense.get_nrow(), A_dense.get_ncol());

	L_dense.get_lower_triangular(A_dense, 0);
	U_dense.get_upper_triangular(A_dense, 0);
	D_dense.get_diagonal(A_dense, 0);*/
	if (A_dense.get_nrow() == 7)
		std::cout << "A dense row = " << A_dense.get_nrow() << std::endl;
	std::unique_ptr<Matrix<T>> U_dense(new Matrix<T>(A_dense.get_nrow(), A_dense.get_ncol()));
	std::unique_ptr<Matrix<T>> L_dense(new Matrix<T>(A_dense.get_nrow(), A_dense.get_ncol()));
	if (A_dense.get_nrow() == 7)
		std::cout << "A dense row = " << A_dense.get_nrow() << std::endl;
	std::unique_ptr<Matrix<T>> D_dense(new Matrix<T>(A_dense.get_nrow(), A_dense.get_ncol()));


	L_dense->get_lower_triangular(A_dense, 0);
	U_dense->get_upper_triangular(A_dense, 0);
	D_dense->get_diagonal(A_dense, 0);
	// Invert elements of D
	D_dense->invert_elements();

	// Convert to CRS
	L.convert_to_crs(*L_dense);
	U.convert_to_crs(*U_dense);
	Dinv.convert_to_crs(*D_dense);

}

template <class T>
void PDE_Poisson<T>::Solve()
{

	//std::cout << "m_v = ";
	//m_v.display();
	//std::cout << "m_f = ";
	//m_f.display();
	//std::cout << "m_Dinv = ";
	//m_Dinv.display_valptr();
	//std::cout << "m_U = ";
	//m_U.display_valptr();
	//// For now it just runs the hybrid GS method but will also run multigrid
	//GaussSeidel(m_Dinv, m_L, m_U, m_v, m_f, m_nGS2, m_nJac, m_nouter);

	//m_v.display();
}

template class PDE_Poisson<float>;
template class PDE_Poisson<double>;
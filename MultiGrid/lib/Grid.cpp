#include "Grid.h"

Grid::Grid() :
	_ndim{ 0 },
	_npoints{ std::vector<size_t>{0} }
{

}

Grid::Grid(size_t _dim, std::vector<size_t> _points) : 
	_ndim{ _dim },
	_npoints{ _points }
{

}

void Grid::restrict(Grid finer_grid)
{
	this->_ndim = finer_grid._ndim;
	if (finer_grid._ndim == 1)
	{
		size_t coarser_npoints = std::ceil(finer_grid.get_totalnpoints() * 0.5);
		this->_npoints = std::vector<size_t>{ coarser_npoints };
	}
		
}

bool Grid::operator==(Grid& const grid)
{
	if (this->_npoints == grid._npoints)
		return true;
	else
		return false;
}

bool Grid::operator!=(Grid& const grid)
{
	if (this->_npoints != grid._npoints)
		return true;
	else
		return false;
}
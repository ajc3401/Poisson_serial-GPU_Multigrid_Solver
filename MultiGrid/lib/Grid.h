#ifndef GRID_H
#define GRID_H
#include <iostream>
#include <vector>

class Grid
{
public:
	Grid();
	Grid(size_t _dim, std::vector<size_t> _points);
	inline size_t get_dim() { return this->_ndim; }
	inline std::vector<size_t> get_npoints() { return this->_npoints; }
	inline size_t get_totalnpoints();

	bool operator==(Grid& const grid);
	bool operator!=(Grid& const grid);

	void restrict(Grid finer_grid);
private:
	size_t _ndim;
	std::vector<size_t> _npoints;
};

size_t Grid::get_totalnpoints()
{
	size_t totalnpoints{ 0 };
	for (size_t i = 0; i < _ndim; i++)
		totalnpoints += _npoints[i];
	return totalnpoints;
}

#endif
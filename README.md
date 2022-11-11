# Poisson_serial-GPU_Multigrid_Solver
A C++/CUDA multigrid solver for Poisson's equation.

## Introduction ##
The main inspiration was to build a multigrid solver<sup>1</sup> with intuitively written code that utilizes efficient sparse matrix-vector products (SpMV) and other simple mathematical operations to perform all calculations.  

As a result, the main solver is built upon a library consisting of a mathematical vector, matrix, and CRS formatted sparse matrix class.  These mathematical object classes are wrappers around a vector class that implemented using CPU operations and one implemented using GPU operations written in CUDA.  Mathematical operations on the CPU are written in custom C++ while those on the GPU are a mixture of custom kernels and the cuSPARSE and cuBLAS libraries.

The smoothing algorithm used was introduced in a recent conference paper<sup>2</sup> as a two-stage iterative variant of Gauss-Seidel (GS2) based on easily parallizable and much faster SpMV operations instead of a triangular solve method.  GS2 composes of an inner Jacobi-Relaxation sweeps that replace the triangular solve.

Finally, although the code was built to solve Poisson's equation, the multigrid solver can be easily adapted to any PDE to be added in the future.

## Building the code ##
In the highest level directory type the following:
```
cmake -S "" -B "CONFIG"
cd CONFIG
cmake --build .
```
where CONFIG is either GPU or Serial depending on whether you want to build the GPU or CPU implementation.

**NOTE:** Only the GPU version is functional at the moment.

## Running the code ##
An example of the interface is given below

```cpp
size_t n_rank{ 32 }; \\ typical variable arising in discussions of multigrid; the number of points is one minus n_rank
size_t n_GS2{ 3 }; \\ number of applications of the GS2 smoothing algorithm
size_t n_Jac{ 2 }; \\ number of inner Jacobi sweeps
size_t n_outer{ 5 }; \\ number of outer sweeps
size_t dim{ 1 }; \\ dimension of grid

float pi = std::numbers::pi_v<float>;

Grid grid(1, std::vector<size_t>{n_rank - 1}); \\ Grid object that hold details of the grid

Vector<float> seed(grid, 0.0f, 2*pi, "sin"); \\ initial condition (here it's a sine wave with period 2pi

size_t n_coarsen{ 2 }; \\ number of times grid is coarsened in the multigrid
float h{ 0.01 }; \\ grid spacing

Multigrid<float,PDE_Poisson<float>> multigrid(h, n_rank, dim, seed, n_coarsen, n_GS2, n_Jac, n_outer); \\ initializes multigrid
multigrid.Solve(); \\ solves multigrid
  ```
## Current TODO ##
1. Fix bug in CPU version


## Future TODO ##
1. Allow flexible choice of smoothing algorithm
2. Add support for 2 and 3D Poisson's equation
3. Miscellaneous performance tuning
4. Add additional PDEs

## References ##
<sup>1</sup>Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). A multigrid tutorial. Society for Industrial and Applied Mathematics.

<sup>2</sup>Thomas, Stephen, Ichitaro Yamazaki, Luc Berger-Vergiat, Brian Kelley, Jonathan Hu,
Paul Mullowney, Sivasankaran Rajamanickam, and Katarzyna Swirydowicz. 2022. TwoStage Gauss-Seidel Preconditioners and Smoothers for Krylov Solvers on a GPU Cluster:
Preprint. Golden, CO: National Renewable Energy Laboratory. NREL/CP-2C00-80263.
https://www.nrel.gov/docs/fy22osti/80263.pdf. 

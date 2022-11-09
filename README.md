# Poisson_serial-GPU_Multigrid_Solver
A C++/CUDA multigrid solver for Poisson's equation.

The main inspiration was to build a multigrid solver with intuitively written code that utilizes efficient sparse matrix-vector products and other simple mathematical operations to perform the entire calculation.  

As a result, the main solver is built upon a library consisting of a mathematical vector, matrix, and CRS formatted sparse matrix class.  These mathematical object classes are wrappers around a vector class that implemented using CPU operations and one implemented using GPU operations written in CUDA.  Mathematical operations on the CPU are written in custom C++ while those on the GPU are a mixture of custom kernels and 

The smoothing algorithm used was introduced in a recent conference paper<sup>1</sup>
## Introduction ##


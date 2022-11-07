#ifndef CUDAVECTOROPERATIONS_H
#define CUDAVECTOROPERATIONS_H
#include <string>

template<class T> void sumVectors(T* a, const T* b, size_t N);
template<class T> void subtractVectors(T* a, const T* b, size_t N);
template<class T> void scalarVectorMultiply(T* a, const T b, size_t N);

template<class T> void dotProduct(const T* a, const T* b, T& c, size_t N);
template<class T> void invertElements(T* a, size_t N);

void applyFunction(double* a, size_t N, std::string function);
void applyFunction(float* a, size_t N, std::string function);
void applyFunction(int* a, size_t N, std::string function);
void applyFunction(size_t* a, size_t N, std::string function);

void l2Norm(float* result, const float* invec, size_t N);
void l2Norm(double* result, const double* invec, size_t N);
void l2Norm(int* result, const int* invec, size_t N);
void l2Norm(size_t* result, const size_t* invec, size_t N);

template<class T> void interpolate1D(T* v_finer, const T* v_coarser, size_t N_coarser);
template<class T> void interject1D(T* v_coarser, const T* v_finer, size_t N_coarser);

template<class T> void setEqual(T* a, const T* b, size_t N);
template<class T> void setNegative(T* a, size_t N);
template<class T> void setValue(T* a, const T b, size_t N);
template<class T> void setRange(size_t left, size_t right, T* a, const T* b);



#endif
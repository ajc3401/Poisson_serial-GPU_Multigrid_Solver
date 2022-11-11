// Reference: https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-

#ifndef CUDAERRORHANDLER_CUH
#define CUDAERRORHANDLER_CUH

#include "cuda_device_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>

// Handles possible errors in CUDA operations.
static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



#endif
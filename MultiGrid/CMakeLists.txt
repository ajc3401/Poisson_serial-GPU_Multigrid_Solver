set(PROJECT_NAME MultiGrid)

################################################################################
# Source groups
################################################################################
set(Headers
    "lib/CRS_Matrix.h"
#    "lib/CudaErrorHandler.cuh"
#    "lib/CudaMatrixOperations.cuh"
#    "lib/CudaMemoryHandler.cuh"
#    "lib/CudaVector.cuh"
#    "lib/CudaVectorOperations.cuh"
    "lib/Grid.h"
    "lib/Matrix.h"
    "lib/SerialMatrixOperations.h"
    "lib/SerialVector.h"
    "lib/SerialVectorOperations.h"
    "lib/Vector.h"
    "lib/VectorBase.h"
    "src/Iterative_Methods.h"
    "src/Multigrid.h"
    "src/PDE_Base.h"
    "src/PDE_Poisson.h"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "lib/CRS_Matrix.cpp"
    "lib/Grid.cpp"
    "lib/Matrix.cpp"
    "lib/SerialMatrixOperations.cpp"
    "lib/SerialVector.cpp"
    "lib/SerialVectorOperations.cpp"
    "lib/Vector.cpp"
    "lib/VectorBase.cpp"
    "src/Iterative_Methods.cpp"
    "src/main.cpp"
    "src/Multigrid.cpp"
    "src/PDE_Poisson.cpp"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${Headers}
    ${Sources}
)

################################################################################
# Target
################################################################################
add_executable(${PROJECT_NAME} ${ALL_FILES})

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
################################################################################
# Includes for CMake from *.props
################################################################################
use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "$ENV{VCTargetsPath}/BuildCustomizations/CUDA 11.7.cmake")

set(ROOT_NAMESPACE MultiGrid)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
)
################################################################################
# Compile definitions
################################################################################
target_compile_definitions(${PROJECT_NAME} PRIVATE
    "$<$<CONFIG:Debug>:"
        "WIN32;"
        "WIN64;"
        "_DEBUG;"
        "_CONSOLE;"
        "__GPU__;"
        "_MBCS"
    ">"
    "$<$<CONFIG:Release>:"
        "WIN32;"
        "WIN64;"
        "NDEBUG;"
        "_CONSOLE;"
        "__GPU__;"
        "_MBCS"
    ">"
    "$<$<CONFIG:Serial>:"
        "WIN32;"
        "WIN64;"
        "_DEBUG;"
        "_CONSOLE;"
        "_MBCS"
    ">"
    "$<$<CONFIG:Serial3>:"
        "WIN32;"
        "WIN64;"
        "_DEBUG;"
        "_CONSOLE;"
        "_MBCS"
    ">"
)

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /Od;
            /W3
        >
        $<$<CONFIG:Release>:
            /O2;
            /Oi;
            /Gy;
            /W3
        >
        $<$<CONFIG:Serial>:
            /Od;
            /W3
        >
        $<$<CONFIG:Serial3>:
            /Od;
            /W3
        >
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
        ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            /DEBUG;
            /SUBSYSTEM:CONSOLE;
            /INCREMENTAL
        >
        $<$<CONFIG:Release>:
            /DEBUG;
            /SUBSYSTEM:CONSOLE;
            /OPT:REF;
            /OPT:ICF
        >
        $<$<CONFIG:Serial>:
            /DEBUG;
            /SUBSYSTEM:CONSOLE;
            /INCREMENTAL
        >
        $<$<CONFIG:Serial2>:
            /DEBUG:FULL
        >
        $<$<CONFIG:Serial3>:
            /DEBUG;
            /SUBSYSTEM:CONSOLE;
            /INCREMENTAL
        >
    )
endif()

################################################################################
# Dependencies
################################################################################
set(ADDITIONAL_LIBRARY_DEPENDENCIES
    "$<$<CONFIG:Debug>:"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32;"
        "cublas;"
        "cusparse"
    ">"
    "$<$<CONFIG:Release>:"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32;"
        "cublas;"
        "cusparse"
    ">"
    "$<$<CONFIG:Serial>:"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32;"
        "cublas;"
        "cusparse"
    ">"
    "$<$<CONFIG:Serial3>:"
        "cudart_static;"
        "kernel32;"
        "user32;"
        "gdi32;"
        "winspool;"
        "comdlg32;"
        "advapi32;"
        "shell32;"
        "ole32;"
        "oleaut32;"
        "uuid;"
        "odbc32;"
        "odbccp32;"
        "cublas;"
        "cusparse"
    ">"
)
target_link_libraries(${PROJECT_NAME} PRIVATE "${ADDITIONAL_LIBRARY_DEPENDENCIES}")


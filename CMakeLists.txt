cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(affinity_test LANGUAGES CXX CUDA)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA libs and includes to link against
find_package(CUDAToolkit REQUIRED)


# CUDA and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(CUDA_FLAGS
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/std:c++17,
    -Xcompiler=-std=c++17 -Wextra>
    $<IF:$<CXX_COMPILER_ID:MSVC>,
    -Xcompiler=/W4,
    -Xcompiler=-Wall>
    $<IF:$<CONFIG:Debug>,
        -G -src-in-ptx,
        -lineinfo>
	--expt-extended-lambda
    -Wno-deprecated-gpu-targets #silence deprecated architectures
    --expt-relaxed-constexpr
    $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
    )


set(CUDA_ARCHS 70)

set(cuda_includes     ${CUDAToolkit_INCLUDE_DIRS})

add_executable(affinity_test affinity_test.cu)
target_include_directories(affinity_test
                      PRIVATE ${cuda_includes})
target_compile_options(affinity_test INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>)
set_target_properties(affinity_test PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})
target_link_libraries(affinity_test CUDA::cuda_driver CUDA::cudart)

# visual studio setup
file(TO_NATIVE_PATH ${cuda_includes} cuda_includes_native)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT affinity_test)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${cuda_includes_native}
    )

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(affinity_test LANGUAGES CXX CUDA)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)

# Find CUDA libs and includes to link against
find_package(CUDAToolkit REQUIRED)

set(cuda_includes     ${CUDAToolkit_INCLUDE_DIRS})

add_executable(affinity_test affinity_test.cu)
target_include_directories(affinity_test
                      PRIVATE ${cuda_includes})
target_link_libraries(affinity_test CUDA::cuda_driver CUDA::cudart_static)

# visual studio setup
file(TO_NATIVE_PATH ${cuda_includes} cuda_includes_native)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT affinity_test)
set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
    $(VC_IncludePath)
    $(WindowsSDK_IncludePath)
    ${cuda_includes_native}
    )

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(affinity_test LANGUAGES CXX HIP)

find_package(hip REQUIRED)

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)


# HIP and C++ compiler flags
set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++17,
        --std=c++17
        >)
set(HIP_FLAGS --std=c++17)
# set(HIP_FLAGS
#     $<IF:$<CXX_COMPILER_ID:MSVC>,
#     -Xcompiler=/std:c++17,
#     -Xcompiler=-std=c++17 -Wextra>
#     $<IF:$<CXX_COMPILER_ID:MSVC>,
#     -Xcompiler=/W4,
#     -Xcompiler=-Wall>
#     $<IF:$<CONFIG:Debug>,
#         -G -src-in-ptx,
#         -lineinfo>
# 	--expt-extended-lambda
#     -Wno-deprecated-gpu-targets #silence deprecated architectures
#     --expt-relaxed-constexpr
#     $<$<CONFIG:Release>:-Xptxas -warn-spills -res-usage>
#     )


# set(HIP_ARCHS 70)

# set(HIP_includes     ${HIPToolkit_INCLUDE_DIRS})

add_executable(affinity_test affinity_test.cu)
set_source_files_properties(affinity_test.cu PROPERTIES LANGUAGE HIP)
target_include_directories(affinity_test
                      PRIVATE ${HIP_includes})
target_compile_options(affinity_test INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>)
# set_target_properties(affinity_test PROPERTIES HIP_ARCHITECTURES ${HIP_ARCHS})
target_link_libraries(affinity_test INTERFACE hip::device INTERFACE hip::host)

# visual studio setup
# file(TO_NATIVE_PATH ${HIP_includes} HIP_includes_native)

# set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT affinity_test)
# set(CMAKE_VS_SDK_INCLUDE_DIRECTORIES
#     $(VC_IncludePath)
#     $(WindowsSDK_IncludePath)
#     ${HIP_includes_native}
#     )

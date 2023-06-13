cmake_minimum_required(VERSION 3.18)

project(
	BRIEF
	VERSION 1.0
	DESCRIPTION "Biomedical Data Compression with ImplicitNeural Function"
	LANGUAGES CXX CUDA
)


set(TCNN_CUDA_ARCHITECTURES "" CACHE STRING "Build tiny-cuda-nn for a specific GPU architecture.")

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()


if (APPLE)
	set(CMAKE_MACOSX_RPATH ON)
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

get_directory_property(TCNN_HAS_PARENT PARENT_DIRECTORY)

# adapted from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

if (DEFINED ENV{TCNN_CUDA_ARCHITECTURES})
	message(STATUS "Obtained target architecture from environment variable TCNN_CUDA_ARCHITECTURES=$ENV{TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES $ENV{TCNN_CUDA_ARCHITECTURES})
elseif (TCNN_CUDA_ARCHITECTURES)
	message(STATUS "Obtained target architecture from CMake variable TCNN_CUDA_ARCHITECTURES=${TCNN_CUDA_ARCHITECTURES}")
	set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
else()
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

# Remove unsupported architectures
list(FILTER CMAKE_CUDA_ARCHITECTURES EXCLUDE REGEX "PTX")
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)

# If the CUDA version does not permit targeting Ampere, don't do so.
if ((80 IN_LIST CMAKE_CUDA_ARCHITECTURES OR 86 IN_LIST CMAKE_CUDA_ARCHITECTURES) AND CUDA_VERSION VERSION_LESS 11.0)
	message(WARNING "CUDA version ${CUDA_VERSION} is too low for targeting Ampere GPUs. Reverting to compute capability 75.")
	list(REMOVE_ITEM CMAKE_CUDA_ARCHITECTURES 80 86)
	if (NOT CMAKE_CUDA_ARCHITECTURES)
		list(APPEND CMAKE_CUDA_ARCHITECTURES 75)
	endif()
endif()

# Sort the list to obtain lowest architecture that must be compiled for.
list(SORT CMAKE_CUDA_ARCHITECTURES COMPARE NATURAL ORDER ASCENDING)
list(GET CMAKE_CUDA_ARCHITECTURES 0 MIN_GPU_ARCH)

string(REPLACE "-virtual" "" MIN_GPU_ARCH "${MIN_GPU_ARCH}")

message(STATUS "Targeting GPU architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if (TCNN_HAS_PARENT)
	set(TCNN_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
	set(TCNN_CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
endif()

if (MIN_GPU_ARCH LESS_EQUAL 70)
	message(WARNING
		"Fully fused MLPs do not support GPU architectures of 70 or less. "
		"Falling back to CUTLASS MLPs. Remove GPU architectures 70 and lower "
		"to allow maximum performance"
	)
endif()

if (CUDA_VERSION VERSION_LESS 10.2)
	message(FATAL_ERROR "CUDA version too low. tiny-cuda-nn require CUDA 10.2 or higher.")
endif()

list(APPEND TCNN_DEFINITIONS -DTCNN_MIN_GPU_ARCH=${MIN_GPU_ARCH})
if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
	# Only compile the shampoo optimizer if
	# a new enough cuBLAS version is available.
	list(APPEND TCNN_DEFINITIONS -DTCNN_SHAMPOO)
endif()

if (TCNN_HAS_PARENT)
	set(TCNN_DEFINITIONS ${TCNN_DEFINITIONS} PARENT_SCOPE)
endif()

if (MSVC)
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
endif()
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

###############################################################################
# Dependencies
###############################################################################
include(FindTIFF)
add_subdirectory(dependencies/tiny-cuda-nn)

###############################################################################
# Include files
###############################################################################
list(APPEND INCLUDE_DIRECTORIES "include")
list(APPEND INCLUDE_DIRECTORIES "dependencies")
list(APPEND INCLUDE_DIRECTORIES ${TIFF_INCLUDE_DIRS})

if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

###############################################################################
# Programe
###############################################################################
add_executable(BRIEF src/main.cu src/common.cu)
target_include_directories(BRIEF PUBLIC ${INCLUDE_DIRECTORIES})
target_link_libraries(BRIEF PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn ${TIFF_LIBRARIES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(BRIEF PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
	target_compile_options(tiny-cuda-nn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
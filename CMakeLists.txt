


LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_PREFIX}")
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

cmake_minimum_required (VERSION 3.0.2)
project (Ustatistics C)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif ()
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)

# Remove 'lib' prefix for shared libraries on Windows
if (WIN32)
  set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()


find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

# ---
# sub-projects.
# core Ustatistics component; builds a static library.
add_subdirectory(core)

add_subdirectory(standardExamplesUstatistics)

# example executable containing the concrete dataset
add_subdirectory(concreteDatasetExample)

# example executable containing a demo app that computes U-statistics on random Gaussian data
add_subdirectory(randomGaussianExample)


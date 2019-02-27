
cmake_minimum_required (VERSION 3.0.2)
project (standardUstatisticsExamples)
include_directories("../core")
find_package(GSL REQUIRED)
add_executable(mean
 mean.c)

target_link_libraries(randomGaussianExample
 Ustatisticslib
 GSL::gsl
 GSL::gslcblas
)

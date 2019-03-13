
cmake_minimum_required (VERSION 3.2.3)
project (randomGaussianExample)

# includes to core U-statistics library.
include_directories("../core")
find_package(GSL REQUIRED)


# executable.
add_executable(randomGaussianExample 
 main.c)

target_link_libraries(randomGaussianExample
 Ustatisticslib # static link with core Ustatistic library.
 GSL::gsl
 GSL::gslcblas
 m
)
set_property(TARGET randomGaussianExample PROPERTY C_STANDARD 11)

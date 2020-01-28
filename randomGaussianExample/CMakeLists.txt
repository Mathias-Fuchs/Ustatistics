
cmake_minimum_required (VERSION 3.2.3)
project (randomGaussianExample)

# includes to core U-statistics library.
include_directories("../core")

#code for regression-specific stuff
include_directories("../regression")

find_package(GSL REQUIRED)

# executable.
add_executable(randomGaussianExample 
 main.c)

target_link_libraries(randomGaussianExample
 Ustatisticslib # static link with core Ustatistic library.
    regressionlib # static link with regression specific code
 GSL::gsl
 GSL::gslcblas
)

if (NOT WIN32) 
 ## only in linux
target_link_libraries(randomGaussianExample
	m
)
endif()



set_property(TARGET randomGaussianExample PROPERTY C_STANDARD 11)





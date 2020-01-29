
cmake_minimum_required (VERSION 3.2.3)
project(regressionlib C)
include_directories("../core")
find_package(GSL REQUIRED)
add_library(regressionlib STATIC
    regressionLearner.c
    supervisedLearning.c
 )
if (NOT WIN32) 
target_link_libraries(regressionlib
	m
)
endif()

target_link_libraries(regressionlib
 Ustatisticslib
 GSL::gsl
 GSL::gslcblas
)
set_property(TARGET regressionlib PROPERTY C_STANDARD 11)

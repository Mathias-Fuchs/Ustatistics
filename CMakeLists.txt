cmake_minimum_required (VERSION 3.0.2)
project (test C)

find_package(GSL REQUIRED)

add_executable(concreteMain 
	regressionLearner.c
	U.c
	concrete.c
)

add_executable(randomMain 
	regressionLearner.c
	U.c
	random.c
)

target_link_libraries(concreteMain
	GSL::gsl
	GSL::gslcblas
)

target_link_libraries(randomMain
	GSL::gsl
	GSL::gslcblas
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slump.dat  ${CMAKE_CURRENT_BINARY_DIR}/slump.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slumpResponse.dat  ${CMAKE_CURRENT_BINARY_DIR}/slumpResponse.dat COPYONLY)
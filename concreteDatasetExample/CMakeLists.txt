
cmake_minimum_required (VERSION 3.0.2)
project (concreteDatasetExample)

# includes to core Ustatistics library.
include_directories("../core")
find_package(GSL REQUIRED)


# executable.
add_executable(concreteDatasetExample
	main.c)

target_link_libraries(concreteDatasetExample
	Ustatisticslib
	GSL::gsl
	GSL::gslcblas
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slump.dat  ${CMAKE_CURRENT_BINARY_DIR}/slump.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slumpResponse.dat  ${CMAKE_CURRENT_BINARY_DIR}/slumpResponse.dat COPYONLY)

message("current source directory")
message(${CMAKE_CURRENT_SOURCE_DIR} )

cmake_minimum_required (VERSION 3.2.3)
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
if (NOT WIN32) 
target_link_libraries(concreteDatasetExample
	# m ## only in linux
)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slump.dat  ${CMAKE_CURRENT_BINARY_DIR}/slump.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slumpResponse.dat  ${CMAKE_CURRENT_BINARY_DIR}/slumpResponse.dat COPYONLY)

message("current source directory")
message(${CMAKE_CURRENT_SOURCE_DIR} )

set_property(TARGET concreteDatasetExample PROPERTY C_STANDARD 11)

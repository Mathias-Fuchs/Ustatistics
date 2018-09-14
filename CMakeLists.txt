# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.0.2)
project (test C)

# set(sqlite3_DIR "../vcpkg/installed/x86-windows/share/sqlite3")
find_package(sqlite3 REQUIRED)
find_package(GSL REQUIRED)

# link_directories("../vcpkg/packages/gsl_x86-windows/lib")

add_executable(main 
	concrete.c
)

target_link_libraries(main
	sqlite3
	GSL::gsl
	GSL::gslcblas
)


configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slump.dat  ${CMAKE_CURRENT_BINARY_DIR}/slump.dat COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/slumpResponse.dat  ${CMAKE_CURRENT_BINARY_DIR}/slumpResponse.dat COPYONLY)

cmake_minimum_required (VERSION 3.2.3)
project (Ustatisticslib C)

# sometimes needed but usually it should be fine
# include_directories("../../vcpkg/installed/x86-windows/include")


find_package(GSL REQUIRED)

# static library.
add_library(Ustatisticslib STATIC
runningStats.c
  binomCoeff.c
U.c
)

# no executable because this is just the core library
if (WIN32)
target_link_libraries(Ustatisticslib
	GSL::gsl
	GSL::gslcblas
)
else()
target_link_libraries(Ustatisticslib
	GSL::gsl
	GSL::gslcblas
	m ## only in linux
)
endif()
set_property(TARGET Ustatisticslib PROPERTY C_STANDARD 11)

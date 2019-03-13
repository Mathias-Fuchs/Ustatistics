
cmake_minimum_required (VERSION 3.2.3)
project (Ustatisticslib)

# sometimes needed but usually it should be fine
# include_directories("../../vcpkg/installed/x86-windows/include")
find_package(GSL REQUIRED)

# static library.
add_library(Ustatisticslib
	regressionLearner.c
	supervisedLearning.c
	U.c
)

# no executable because this is just the core library
target_link_libraries(Ustatisticslib
	GSL::gsl
	GSL::gslcblas
)
set_property(TARGET Ustatisticslib PROPERTY C_STANDARD 11)


cmake_minimum_required (VERSION 3.2.3)
project (mean)
project(variance)

include_directories("../core")
find_package(GSL REQUIRED)

add_executable(mean
 mean.c)

add_executable(variance
 variance.c)

add_executable(centralFourthMoment
 centralFourthMoment.c)


if (NOT WIN32) 
target_link_libraries(mean
	m
)
target_link_libraries(variance
	m
)
target_link_libraries(centralFourthMoment
	m
)
endif()

target_link_libraries(mean
 Ustatisticslib
 GSL::gsl
 GSL::gslcblas
)

target_link_libraries(variance
 Ustatisticslib
 GSL::gsl
 GSL::gslcblas
)

target_link_libraries(centralFourthMoment
  Ustatisticslib
 GSL::gsl
 GSL::gslcblas
)
set_property(TARGET mean PROPERTY C_STANDARD 11)
set_property(TARGET variance PROPERTY C_STANDARD 11)
set_property(TARGET centralFourthMoment PROPERTY C_STANDARD 11)

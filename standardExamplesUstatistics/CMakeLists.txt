
cmake_minimum_required (VERSION 3.0.2)
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

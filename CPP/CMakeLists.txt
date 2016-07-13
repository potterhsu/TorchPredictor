cmake_minimum_required(VERSION 3.2)
project(Build)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

find_package(OpenCV REQUIRED)

SET(OPENBLAS_HOME $ENV{OPENBLAS_HOME})
IF(NOT OPENBLAS_HOME)
    MESSAGE(FATAL_ERROR "Environment variable OPENBLAS_HOME is undefined.")
ENDIF()
include_directories("${OPENBLAS_HOME}/include")
link_directories("${OPENBLAS_HOME}/lib")

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(sample)
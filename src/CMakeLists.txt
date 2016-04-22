cmake_minimum_required(VERSION 2.4.6)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fopenmp -march=native")

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  add_definitions(-DCOMPILEDWITHC11)
  message(STATUS "Using flag -std=c++11")
endif()

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
add_definitions(-DBOOST_UBLAS_NDEBUG)
# list(APPEND CMAKE_MODULE_PATH 
find_package(OpenCV REQUIRED)
#find_package(Boost REQUIRED)

set(LIBS
  ${OpenCV_LIBS}
#  ${Boost_LIBRARIES}
  )

add_executable(x_corner main.cpp)# dbscan.cpp)
target_link_libraries(x_corner ${LIBS})
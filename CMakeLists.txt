cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Werror -pedantic")

project(hdcalib)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_Include_Dirs})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

add_library(hdcalib hdcalib.cpp)

include_directories("/usr/local/include/hdmarker")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

target_link_libraries(hdcalib
  hdmarker
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${RUNNINGSTATS_LIBRARIES}
  ${CERES_LIBRARIES}
  raw
)





add_executable(hdcalib-tool "main.cpp")
target_link_libraries(hdcalib-tool hdcalib)

find_package(GTest REQUIRED)
add_executable(tests tests.cpp)
target_link_libraries(tests
  hdmarker
  hdcalib
  ${GTEST_BOTH_LIBRARIES}
  )

add_executable(valgrind-tests valgrind-tests.cpp)
target_link_libraries(valgrind-tests
  hdmarker
  hdcalib
  ${GTEST_BOTH_LIBRARIES}
)

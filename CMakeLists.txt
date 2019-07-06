cmake_minimum_required(VERSION 2.8)

project(hdcalib)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Werror -pedantic")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

add_library(hdcalib hdcalib.cpp cornerstore.cpp)

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

add_executable(hdcalib-analyze-lf "analyze-lf.cpp")

add_executable(extract-green "extract-green.cpp")

add_executable(hdcalib-localize-exception "localize-exception.cpp")

add_executable(hdcalib-marker-detection "marker-detection.cpp")

add_executable(hdcalib-test-submarker-detection "test-submarker-detection.cpp")

foreach(program hdcalib-tool extract-green hdcalib-localize-exception hdcalib-analyze-lf hdcalib-marker-detection)
    target_link_libraries(${program} hdcalib)
endforeach()

find_package(GTest REQUIRED)
add_executable(tests tests.cpp)

foreach(program tests hdcalib-test-submarker-detection)
  target_link_libraries(${program}
    hdcalib
    ${GTEST_BOTH_LIBRARIES}
  )
endforeach()

add_executable(valgrind-tests valgrind-tests.cpp)
target_link_libraries(valgrind-tests
  hdmarker
  hdcalib
  ${GTEST_BOTH_LIBRARIES}
)

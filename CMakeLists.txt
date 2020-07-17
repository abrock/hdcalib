cmake_minimum_required(VERSION 2.8)

project(hdcalib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -Werror -pedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -pedantic")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

add_library(cornerstore cornerstore.cpp)
add_library(calib-stats hdcalib-stats.cpp)
add_library(hdcalib-readwrite hdcalib-readwrite.cpp)
add_library(hdcalib-ceres-calib hdcalib-ceres-calib.cpp)
add_library(hdcalib-analyze-grid hdcalib-analyze-grid.cpp)
add_library(hdcalib-rectify hdcalib-rectify.cpp)
add_library(hdcalib-lensfun hdcalib-lensfun.cpp)
add_library(hdcalib hdcalib.cpp)
add_library(hdcalib-cornercolor cornercolor.cpp)
include_directories("/usr/local/include/hdmarker")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

find_package(catlogger REQUIRED)
include_directories(${CATLOGGER_INCLUDE_DIRS})

target_link_libraries(cornerstore hdcalib)
target_link_libraries(hdcalib-readwrite
    calib-stats
    hdmarker
    )

target_link_libraries(hdcalib-rectify
    hdcalib-ceres-calib
    hdcalib
    ${CERES_LIBRARIES}
)

target_link_libraries(hdcalib-analyze-grid
    hdcalib-rectify
)

target_link_libraries(hdcalib
  cornerstore
  calib-stats
  hdcalib-readwrite
  hdcalib-ceres-calib
  hdcalib-cornercolor
  hdcalib-analyze-grid
  hdcalib-rectify
  hdmarker
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_IOSTREAMS_LIBRARY}
  ${RUNNINGSTATS_LIBRARIES}
  ${CATLOGGER_LIBRARIES}
  raw
)

add_executable(hdcalib-tool "main.cpp")

add_executable(hdcalib-overexposed hdcalib-overexposed.cpp)

add_executable(hdcalib-analyze-lf "analyze-lf.cpp")

add_executable(extract-green "extract-green.cpp")

add_executable(hdcalib-localize-exception "localize-exception.cpp")

add_executable(hdcalib-marker-detection "marker-detection.cpp")

add_executable(hdcalib-offset-comparison hdcalib-offset-comparison.cpp)

add_executable(hdcalib-test-submarker-detection "test-submarker-detection.cpp")

add_executable(hdcalib-speedtest-prepare-calibration "speedtest-prepare-calibration.cpp")

add_executable(hdcalib-simulate "simulate.cpp")

add_executable(hdcalib-rectify-tool rectify.cpp)

add_executable(hdcalib-analyze-detection-rates analyze-detection-rates.cpp)

foreach(program
        hdcalib-tool
        extract-green
        hdcalib-localize-exception
        hdcalib-analyze-detection-rates
        hdcalib-analyze-lf
        hdcalib-marker-detection
        hdcalib-offset-comparison
        hdcalib-speedtest-prepare-calibration
        hdcalib-simulate
        hdcalib-rectify-tool
    )
    target_link_libraries(${program} hdcalib)
endforeach()

foreach(program
        hdcalib-analyze-detection-rates
        hdcalib-offset-comparison
        hdcalib-tool
        hdcalib-speedtest-prepare-calibration
        hdcalib-simulate
        hdcalib-rectify-tool
        )
    target_link_libraries(${program} ParallelTime)
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

add_executable(hdcalib-compare-images compare-images.cpp)

target_link_libraries(hdcalib-compare-images
      ${OpenCV_LIBS}
      ${RUNNINGSTATS_LIBRARIES}
    )

target_link_libraries(hdcalib-overexposed
      ${OpenCV_LIBS}
      ${RUNNINGSTATS_LIBRARIES}
    )

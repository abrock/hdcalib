cmake_minimum_required(VERSION 2.8)

project(hdcalib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fext-numeric-literals -Wall -Werror -pedantic")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -pedantic")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

add_library(hdcalib
    cornercache.cpp
    cornercolor.cpp
    cornerstore.cpp
    fitgrid-lib.cpp
    griddescription.cpp
    hdcalib-spline-calib.cpp
    hdcalib-analyze-grid.cpp
    hdcalib-ceres-calib.cpp
    hdcalib-lensfun.cpp
    hdcalib-readwrite.cpp
    hdcalib-rectify.cpp
    hdcalib-stats.cpp
    hdcalib.cpp
    similarity2d.cpp
    )

include_directories("/usr/local/include/hdmarker")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(RunningStats REQUIRED)
include_directories(${RUNNINGSTATS_INCLUDE_DIRS})

find_package(catlogger REQUIRED)
include_directories(${CATLOGGER_INCLUDE_DIRS})

target_link_libraries(hdcalib
  hdmarker
  ParallelTime
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

add_executable(hdcalib-fitgrid fit-grid.cpp)

add_executable(hdcalib-yaml2bin yaml2bin.cpp)

add_executable(hdcalib-analyze-detection-rates analyze-detection-rates.cpp)

add_executable(raw2png raw2png.cpp)

add_executable(test-similarity2d test-similarity2d.cpp)

add_executable(test-spline-model test-spline-model.cpp)
target_link_libraries(test-spline-model
    ${CERES_LIBRARIES}
    )

add_executable(hdcalib-2dgauss 2dgauss.cpp)

target_link_libraries(hdcalib-2dgauss
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
#    ${Boost_FILESYSTEM_LIBRARY}
#    ${Boost_SYSTEM_LIBRARY}
#    ${Boost_IOSTREAMS_LIBRARY}
    ${RUNNINGSTATS_LIBRARIES}
#    ${CATLOGGER_LIBRARIES}
    )

foreach(program
        hdcalib-tool
        extract-green
        hdcalib-localize-exception
        hdcalib-fitgrid
        hdcalib-analyze-detection-rates
        hdcalib-analyze-lf
        hdcalib-marker-detection
        hdcalib-offset-comparison
        hdcalib-speedtest-prepare-calibration
        hdcalib-simulate
        hdcalib-rectify-tool
        hdcalib-yaml2bin
        raw2png
        test-similarity2d
    )
    target_link_libraries(${program} hdcalib)
endforeach()

find_package(GTest REQUIRED)
add_executable(tests tests.cpp)

foreach(program tests hdcalib-test-submarker-detection test-similarity2d test-spline-model)
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

add_executable(hdcalib-check-corner-cache check-corner-cache.cpp)

target_link_libraries(hdcalib-check-corner-cache
      ${OpenCV_LIBS}
      hdmarker
      ${Boost_FILESYSTEM_LIBRARY}
      ${Boost_SYSTEM_LIBRARY}
      ${Boost_IOSTREAMS_LIBRARY}
    )

add_executable(marker-fit-experiment marker-fit-experiment.cpp)

target_link_libraries(marker-fit-experiment
    ${OpenCV_LIBS}
    ${RUNNINGSTATS_LIBRARIES}
    ${CERES_LIBRARIES}
    hdmarker
    ParallelTime
    )

add_executable(hdcalib-analyze-video analyze-video.cpp)
target_link_libraries(hdcalib-analyze-video
    ${OpenCV_LIBS}
    )

add_executable(hdcalib-image-series-noise image-series-noise.cpp)
target_link_libraries(hdcalib-image-series-noise
    ${OpenCV_LIBS}
    ${RUNNINGSTATS_LIBRARIES}
    )

add_library(plotoptflow libplotoptflow.cpp libfitspline.cpp)
target_link_libraries(plotoptflow PUBLIC
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    )

add_executable(hdcalib-plot-flow plot-flow.cpp)
target_link_libraries(hdcalib-plot-flow PUBLIC
    plotoptflow
    ${RUNNINGSTATS_LIBRARIES}
    )

add_executable(hdcalib-add-noise add-noise.cpp)
target_link_libraries(hdcalib-add-noise PUBLIC
    libdownscale
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    )

add_executable(hdcalib-fill-black hdcalib-fill-black.cpp)
target_link_libraries(hdcalib-fill-black PUBLIC
    ${OpenCV_LIBS}
    hdcalib
    #${Boost_FILESYSTEM_LIBRARY}
    #${Boost_SYSTEM_LIBRARY}
    #${Boost_IOSTREAMS_LIBRARY}

    )

add_executable(radial-models radial-models.cpp)
target_link_libraries(radial-models
    ${CERES_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    )

add_executable(spline-model spline-model.cpp)
target_link_libraries(spline-model
    ${CERES_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    )


add_library(libdownscale libdownscale.cpp)
target_link_libraries(libdownscale
    ${OpenCV_LIBS}
    )

add_executable(hdcalib-downscale downscale.cpp)
target_link_libraries(hdcalib-downscale
    libdownscale
    )

add_executable(hdcalib-scan-analysis scan-analysis.cpp)
target_link_libraries(hdcalib-scan-analysis
    ${OpenCV_LIBS}
    hdcalib
    )

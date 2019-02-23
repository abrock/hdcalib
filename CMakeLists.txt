cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

project(hdcalib)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_Include_Dirs})

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

add_library(hdcalib calib.cpp)

target_link_libraries(hdcalib
  hdmarker
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${OpenCV_LIBS}
  raw
)


add_executable(hdcalib-tool "main.cpp")
target_link_libraries(hdcalib-tool hdcalib)

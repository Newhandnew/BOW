cmake_minimum_required(VERSION 2.8)
project(bow)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(bow bow.cpp)
add_executable(compareBOF compareBOF.cpp)
target_Link_Libraries(bow ${OpenCV_LIBS})
target_Link_Libraries(compareBOF ${OpenCV_LIBS})

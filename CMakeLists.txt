cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

project(lab2 LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} main.cu)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} 
                                                    ../include/
                                                    ../lib/stb/)
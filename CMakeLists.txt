cmake_minimum_required(VERSION 3.17)
project(hello_world LANGUAGES CXX CUDA) # Use CUDA

# Main
file(GLOB MAIN_SRC "${PROJECT_SOURCE_DIR}/*.cu")
list(APPEND PROJECT_SRC ${MAIN_SRC})

# OpenCV
find_package(OpenCV REQUIRED)
list(APPEND PROJECT_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND PROJECT_LIBS ${OpenCV_LIBS})

# Program
include_directories(${PROJECT_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${PROJECT_SRC})
target_link_libraries(${PROJECT_NAME} ${PROJECT_LIBS})
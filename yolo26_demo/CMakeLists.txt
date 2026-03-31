cmake_minimum_required(VERSION 3.10)
project(yolo26_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AXCL library
if(NOT AXCL_DIR)
    set(AXCL_DIR "/usr" CACHE PATH "AXCL installation directory")
endif()
include_directories(${AXCL_DIR}/include/axcl)
link_directories(${AXCL_DIR}/lib/axcl)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
    axcl/yolo26.cc
    axcl/ax_model_runner/ax_model_runner_axcl.cpp
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/axcl
    ${CMAKE_CURRENT_SOURCE_DIR}/axcl/ax_model_runner
    ${CMAKE_CURRENT_SOURCE_DIR}/base
    ${CMAKE_CURRENT_SOURCE_DIR}/utilities
)

# Executable
add_executable(yolo26 ${SOURCES})
target_link_libraries(yolo26
    axcl_rt
    ${OpenCV_LIBS}
)

# Parallel inference test
add_executable(parallel_test parallel_inference_test.cpp axcl/ax_model_runner/ax_model_runner_axcl.cpp)
target_link_libraries(parallel_test
    axcl_rt
    ${OpenCV_LIBS}
    pthread
)

# Context type test
add_executable(context_test context_test.cpp axcl/ax_model_runner/ax_model_runner_axcl.cpp)
target_link_libraries(context_test
    axcl_rt
    ${OpenCV_LIBS}
)

# tools/CMakeLists.txt
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${AXCL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/yolo26_demo/axcl
    ${CMAKE_SOURCE_DIR}/yolo26_demo/axcl/ax_model_runner
    ${CMAKE_SOURCE_DIR}/yolo26_demo/base
    ${CMAKE_SOURCE_DIR}/yolo26_demo
    ${CMAKE_SOURCE_DIR}/yolo26_demo/utilities
)

# NPU benchmark tool — comprehensive AX650N performance testing
add_executable(npu_bench npu_bench.cpp
    ${CMAKE_SOURCE_DIR}/yolo26_demo/axcl/ax_model_runner/ax_model_runner_axcl.cpp
)
target_link_libraries(npu_bench
    app_inference
    ${AXCL_LIBRARIES}
    ${OpenCV_LIBS}
    Threads::Threads
)


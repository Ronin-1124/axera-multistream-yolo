# tests/unit CMakeLists.txt
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── No-external-deps unit tests ───────────────────────────────────────────────
macro(add_unit_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} Threads::Threads)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

add_unit_test(test_ringbuffer
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ringbuffer.cpp
)

add_unit_test(test_thread_safe_queue
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_safe_queue.cpp
)

add_unit_test(test_postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/test_postprocess.cpp
)

add_unit_test(test_config
    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.cpp
)
target_link_libraries(test_config app_config)

# ── OpenCV-based unit tests ───────────────────────────────────────────────────
macro(add_opencv_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} ${OpenCV_LIBS} Threads::Threads)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

add_opencv_test(test_mosaic_renderer
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mosaic_renderer.cpp
)
target_link_libraries(test_mosaic_renderer app_mosaic app_postprocess)

# ── Tests requiring AXCL hardware (not registered with ctest by default) ────
macro(add_axcl_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} ${AXCL_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)
    target_link_options(${name} PRIVATE -Wl,-rpath,/usr/lib/axcl -Wl,-rpath,/usr/lib/axcl/ffmpeg)
    # Excluded from automated ctest runs (requires AX650N hardware)
    # add_test(NAME ${name} COMMAND ${name})
endmacro()

add_axcl_test(test_axcl_context
    ${CMAKE_CURRENT_SOURCE_DIR}/test_axcl_context.cpp
)
target_link_libraries(test_axcl_context app_postprocess)

add_axcl_test(test_inference_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inference_engine.cpp
)
target_link_libraries(test_inference_engine app_inference)

# ── FFmpeg-based tests (no hardware required, not registered with ctest) ──────
macro(add_ffmpeg_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} ${AXCL_FFMPEG_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)
    target_link_options(${name} PRIVATE -Wl,-rpath,/usr/lib/axcl/ffmpeg)
    # Excluded from automated ctest runs (requires FFmpeg libraries)
    # add_test(NAME ${name} COMMAND ${name})
endmacro()

add_ffmpeg_test(test_stream_source
    ${CMAKE_CURRENT_SOURCE_DIR}/test_stream_source.cpp
)
target_link_libraries(test_stream_source app_stream)

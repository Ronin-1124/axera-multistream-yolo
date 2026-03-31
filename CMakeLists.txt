cmake_minimum_required(VERSION 3.10)
project(axera-multistream-yolo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build unit tests" ON)

find_package(Threads REQUIRED)

# AXCL
if(NOT AXCL_DIR)
    set(AXCL_DIR "/usr" CACHE PATH "AXCL installation directory")
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindAXCL)
include(${CMAKE_SOURCE_DIR}/cmake/CompilerOptions.cmake)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Dependencies
include(dependencies)

# FFmpeg headers
include_directories(${AXCL_FFMPEG_INCLUDE_DIRS})

# Include path for all app headers
set(APP_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include ${AXCL_INCLUDE_DIR} ${AXCL_FFMPEG_INCLUDE_DIRS})

# ── Static libraries ──────────────────────────────────────────────────────────
# libconfig: config parsing, no external deps
add_library(app_config STATIC
    src/config.cpp
)
target_include_directories(app_config PUBLIC ${CMAKE_SOURCE_DIR}/include)

# libmosaic: mosaic renderer, OpenCV only
add_library(app_mosaic STATIC
    src/mosaic_renderer.cpp
)
target_include_directories(app_mosaic PUBLIC ${CMAKE_SOURCE_DIR}/include ${AXCL_INCLUDE_DIR})
target_link_libraries(app_mosaic ${OpenCV_LIBS})

# libpostprocess: YOLO postprocessing (depends on axcl_model.cpp for types)
add_library(app_postprocess STATIC
    src/postprocess.cpp
    src/axcl_model.cpp
)
target_include_directories(app_postprocess PUBLIC ${CMAKE_SOURCE_DIR}/include ${AXCL_INCLUDE_DIR})
target_link_libraries(app_postprocess ${OpenCV_LIBS})

# libstream: stream source (FFmpeg)
add_library(app_stream STATIC
    src/stream_source.cpp
)
target_include_directories(app_stream PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(app_stream ${AXCL_FFMPEG_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)

# libinference: inference engine (AXCL + postprocess)
add_library(app_inference STATIC
    src/inference_engine.cpp
)
target_include_directories(app_inference PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(app_inference app_postprocess ${AXCL_LIBRARIES} ${OpenCV_LIBS})

# libcollector: result collector (depends on mosaic + postprocess)
add_library(app_collector STATIC
    src/result_collector.cpp
)
target_include_directories(app_collector PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(app_collector app_mosaic app_postprocess Threads::Threads)

# libpipeline: pipeline orchestrator (depends on all)
add_library(app_pipeline STATIC
    src/pipeline.cpp
)
target_include_directories(app_pipeline PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(app_pipeline app_collector app_inference app_stream app_postprocess app_mosaic app_config ${OpenCV_LIBS} Threads::Threads)

# ── Main executable ───────────────────────────────────────────────────────────
add_executable(axera_multistream_yolo src/main.cpp)
target_include_directories(axera_multistream_yolo PRIVATE ${APP_INCLUDE_DIRS})
target_link_libraries(axera_multistream_yolo app_pipeline ${AXCL_LIBRARIES} ${OpenCV_LIBS} Threads::Threads)
target_link_options(axera_multistream_yolo PRIVATE -Wl,-rpath,/usr/lib/axcl -Wl,-rpath,/usr/lib/axcl/ffmpeg)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/unit)
endif()

# Tools
add_subdirectory(tools)

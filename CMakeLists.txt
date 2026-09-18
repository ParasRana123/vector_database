cmake_minimum_required(VERSION 3.20)

project(VectorDB VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
)

# MSVC specific configuration
if(MSVC)
    add_compile_options(/FS)
endif()

# Core VectorDB Library
set(VECTORDB_SOURCES
    src/core/distance.cpp
    src/core/thread_pool.cpp
    src/core/metadata.cpp
    src/index/flat_index.cpp
    src/index/hnsw_index.cpp
    src/storage/serializer.cpp
    src/storage/wal.cpp
    src/embedding/embedder.cpp
    src/engine/collection.cpp
    src/engine/database.cpp
)

add_library(vectordb_core STATIC
    ${VECTORDB_SOURCES}
)

if(MSVC)
    target_compile_options(vectordb_core PRIVATE /W3 /O2)
else()
    target_compile_options(vectordb_core PRIVATE -O3 -Wall -Wextra)
endif()

target_include_directories(vectordb_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(vectordb_core PUBLIC
    Threads::Threads
)

if(WIN32)
    target_link_libraries(vectordb_core PUBLIC "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
else()
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
        target_link_libraries(vectordb_core PUBLIC "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    else()
        target_link_libraries(vectordb_core PUBLIC onnxruntime)
    endif()
endif()

if(MSVC)
    set_property(TARGET vectordb_core PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# Macro to configure executable targets
macro(add_vectordb_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE vectordb_core)
    if(MSVC)
        set_property(TARGET ${target_name} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
    if(WIN32 AND EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:${target_name}>/onnxruntime.dll"
        )
    endif()
endmacro()

# CLI and Server Executables
add_vectordb_executable(vectordb_cli src/cli/main_cli.cpp)
add_vectordb_executable(vectordb_server src/server/http_server.cpp)
if(WIN32)
    target_link_libraries(vectordb_server PRIVATE ws2_32)
endif()

# Test Suite
add_vectordb_executable(test_distance tests/test_distance.cpp)
add_vectordb_executable(test_metadata tests/test_metadata.cpp)
add_vectordb_executable(test_flat_index tests/test_flat_index.cpp)
add_vectordb_executable(test_hnsw_index tests/test_hnsw_index.cpp)
add_vectordb_executable(test_database tests/test_database.cpp)
add_vectordb_executable(test_benchmark tests/test_benchmark.cpp)

if(EXISTS "${CMAKE_SOURCE_DIR}/examples/inspect_model.cpp")
    add_vectordb_executable(inspect_model examples/inspect_model.cpp)
endif()

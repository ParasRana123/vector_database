cmake_minimum_required(VERSION 3.20)

project(VectorDB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
)

add_executable(test_onnx
    src/test_onnx.cpp
)

target_include_directories(test_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(test_onnx PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)

add_executable(inspect_model
    examples/inspect_model.cpp
)

target_include_directories(inspect_model PRIVATE
    "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(inspect_model PRIVATE
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
)
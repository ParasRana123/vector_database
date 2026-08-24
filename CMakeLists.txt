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

target_compile_definitions(test_onnx PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Ensure the executable loads the DLL that matches the headers and import library.
add_custom_command(TARGET test_onnx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "$<TARGET_FILE_DIR:test_onnx>/onnxruntime.dll"
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

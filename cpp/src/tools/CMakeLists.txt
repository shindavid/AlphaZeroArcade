# cpp/src/tools/CMakeLists.txt
# -----------------------------
# Build the onnx‐to‐TensorRT‐plan compiler

# 1. Give it a name:
add_executable(onnx_engine_builder
    OnnxEngineBuilder.cpp
)

# 2. Make sure it can find the headers:
target_include_directories(onnx_engine_builder PRIVATE
    /usr/include/x86_64-linux-gnu           # TensorRT headers live here
    /usr/local/cuda/include                 # CUDA headers
)

# 3. Link in TensorRT, the ONNX parser, plugins, and CUDA runtime:
target_link_libraries(onnx_engine_builder PRIVATE
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
)

# 5. Drop the binary into bin/tools so it lives at
#    ${CMAKE_BINARY_DIR}/bin/tools/onnx_engine_builder
set_target_properties(onnx_engine_builder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tools"
)

# 6. Record it for metadata:
append_target_metadata("tools" onnx_engine_builder)

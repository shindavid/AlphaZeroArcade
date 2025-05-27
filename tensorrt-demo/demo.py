#!/usr/bin/env python3
import subprocess

print("Generating ONNX model...")
subprocess.run(["./generate_model.py"], check=True)

print("Compiling test_trt...")
compile_cmd = """
g++ test_trt.cpp -o test_trt \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lcudart \
  -std=c++17 -O2
  """
subprocess.run(compile_cmd, shell=True, check=True)

print("Compiling build_engine...")
compile_cmd2 = """
g++ build_engine.cpp -o build_engine \
  -I/usr/include/x86_64-linux-gnu -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lnvinfer_plugin -lcudart \
  -std=c++17 -O2
"""
subprocess.run(compile_cmd2, shell=True, check=True)

print("Building TensorRT engine from ONNX model...")
build_cmd = "./build_engine model.onnx engine.plan 32"
subprocess.run(build_cmd, shell=True, check=True)

print("Running test_trt with the generated TensorRT engine...")
run_cmd = "./test_trt engine.plan"
subprocess.run(run_cmd, shell=True, check=True)

print('Success!')

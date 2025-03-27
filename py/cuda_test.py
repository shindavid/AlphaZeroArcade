#!/usr/bin/env python3

import torch
import time

# Ensure GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Running on: {device.upper()}")

# Define matrix size (higher = more compute-heavy)
MATRIX_SIZE = 8192

# Allocate random matrices
A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

# Warm-up (ensures stable GPU performance)
for _ in range(10):
    _ = A @ B

# Benchmark GPU
torch.cuda.synchronize()
start_time = time.time()
for _ in range(10):
    _ = A @ B
torch.cuda.synchronize()
gpu_time = time.time() - start_time

print(f"🚀 GPU Time for 10 matrix multiplications: {gpu_time:.4f} seconds")

# Benchmark CPU
A_cpu = A.cpu()
B_cpu = B.cpu()

start_time = time.time()
for _ in range(10):
    _ = A_cpu @ B_cpu
cpu_time = time.time() - start_time

print(f"🐌 CPU Time for 10 matrix multiplications: {cpu_time:.4f} seconds")

speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
print(f"⚡ Speedup (CPU vs. GPU): {speedup:.2f}x")

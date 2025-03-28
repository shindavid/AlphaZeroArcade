#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Benchmark: Raw Matrix Multiplication
# ----------------------------
def benchmark_matrix_mult():
    N = 8192
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    # Warm up GPU.
    for _ in range(5):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    # Time 10 iterations of matrix multiplication.
    start = time.time()
    for _ in range(10):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    print(f"Matrix multiplication: Total time for 10 runs: {total_time:.4f} sec, "
          f"avg per run: {total_time/10:.4f} sec")

# ----------------------------
# Benchmark: Complex Training Loop
# ----------------------------
class ComplexModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=1024, num_layers=10, num_classes=10):
        super(ComplexModel, self).__init__()
        layers = []
        # First block: Linear + BatchNorm + ReLU.
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU(inplace=True))
        # Add additional blocks.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc_out(x)
        return x

def benchmark_complex_training_loop(num_iters=100):
    # Model parameters.
    batch_size = 64
    input_size = 1024
    num_classes = 10

    # Initialize the complex model.
    model = ComplexModel(input_size=input_size, num_layers=10, num_classes=num_classes).to(device)
    model.train()

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy input and target.
    dummy_input = torch.randn(batch_size, input_size, device=device)
    dummy_target = torch.randint(0, num_classes, (batch_size,), device=device)

    # Warm-up iterations to mitigate startup overhead.
    for _ in range(10):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Benchmark the full training iteration (forward + backward + optimizer step)
    start = time.time()
    for _ in range(num_iters):
        optimizer.zero_grad()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    avg_time = total_time / num_iters
    print(f"Complex training loop: Total time for {num_iters} iterations: {total_time:.4f} sec, "
          f"avg per iteration: {avg_time:.4f} sec")

if __name__ == "__main__":
    print("Benchmarking raw matrix multiplication...")
    benchmark_matrix_mult()

    print("\nBenchmarking complex training loop...")
    benchmark_complex_training_loop(num_iters=100)

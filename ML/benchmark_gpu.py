import torch
import time
import statistics

# -------------------------------------------------
# Helper: run a function N times and synchronize MPS
# -------------------------------------------------
def run_benchmark(fn, device, iters=10):
    times = []
    for _ in range(iters):
        start = time.time()
        fn()
        if device == "mps":
            torch.mps.synchronize()
        end = time.time()
        times.append(end - start)
    return statistics.mean(times), statistics.stdev(times)


# -------------------------------------------------
# Matrix multiply benchmark
# -------------------------------------------------
def benchmark_matmul(device, size=4096):
    print(f"\n[Matmul Benchmark] Device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm-up
    for _ in range(5):
        torch.matmul(a, b)
        if device == "mps":
            torch.mps.synchronize()

    def op():
        torch.matmul(a, b)

    mean, std = run_benchmark(op, device)
    print(f"Avg Time: {mean:.5f} s  |  Std: {std:.5f} s")


# -------------------------------------------------
# Convolution benchmark
# -------------------------------------------------
def benchmark_conv(device, batch=64, channels=3, size=256):
    print(f"\n[Conv Benchmark] Device: {device}")

    x = torch.randn(batch, channels, size, size, device=device)
    conv = torch.nn.Conv2d(channels, 64, kernel_size=3, padding=1).to(device)

    # Warm-up
    for _ in range(5):
        conv(x)
        if device == "mps":
            torch.mps.synchronize()

    def op():
        conv(x)

    mean, std = run_benchmark(op, device)
    print(f"Avg Time: {mean:.5f} s  |  Std: {std:.5f} s")


# -------------------------------------------------
# MLP forward + backward benchmark
# -------------------------------------------------
def benchmark_nn(device):
    print(f"\n[NN Forward+Backward Benchmark] Device: {device}")

    model = torch.nn.Sequential(
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(),
        torch.nn.Linear(4096, 4096),
    ).to(device)

    x = torch.randn(64, 4096, device=device)
    y = torch.randn(64, 4096, device=device)

    # Warm-up
    for _ in range(5):
        out = model(x)
        loss = (out - y).pow(2).mean()
        loss.backward()
        model.zero_grad()
        if device == "mps":
            torch.mps.synchronize()

    def op():
        out = model(x)
        loss = (out - y).pow(2).mean()
        loss.backward()
        model.zero_grad()

    mean, std = run_benchmark(op, device)
    print(f"Avg Time: {mean:.5f} s  |  Std: {std:.5f} s")


# -------------------------------------------------
# Run all benchmarks
# -------------------------------------------------
print("\n===== PyTorch Device Benchmark (10 runs, warmed up) =====")
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# CPU
benchmark_matmul("cpu")
benchmark_conv("cpu")
benchmark_nn("cpu")

# MPS
if torch.backends.mps.is_available():
    benchmark_matmul("mps")
    benchmark_conv("mps")
    benchmark_nn("mps")
else:
    print("\nMPS not available on this machine.")


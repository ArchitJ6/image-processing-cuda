import pandas as pd
import matplotlib.pyplot as plt

# Read benchmark data
df = pd.read_csv("output/benchmark.csv")

# Get latest entry for each filter
latest = (
    df.groupby("filter")
      .tail(1)
      .copy()
)

# Desired filter order
order = [
    "grayscale",
    "blur",
    "sobel",
    "sharpen"
]

latest["filter"] = pd.Categorical(
    latest["filter"],
    categories=order,
    ordered=True
)

latest = latest.sort_values("filter")

# ==========================
# GPU vs CPU chart
# ==========================
plt.figure(figsize=(10, 6))

x = range(len(latest))

plt.bar(
    [i - 0.2 for i in x],
    latest["gpu_ms"],
    width=0.4,
    label="GPU"
)

plt.bar(
    [i + 0.2 for i in x],
    latest["cpu_ms"],
    width=0.4,
    label="CPU"
)

plt.xticks(x, latest["filter"])

plt.ylabel("Time (ms)")
plt.xlabel("Filter")
plt.title("CUDA vs CPU Performance")

plt.legend()
plt.tight_layout()

plt.savefig(
    "output/benchmark.png",
    dpi=300
)

print("Generated output/benchmark.png")

# ==========================
# Speedup chart
# ==========================
plt.figure(figsize=(10, 6))

plt.bar(
    latest["filter"],
    latest["speedup"]
)

plt.ylabel("Speedup (x)")
plt.xlabel("Filter")
plt.title("CUDA Speedup over CPU")

plt.tight_layout()

plt.savefig(
    "output/speedup.png",
    dpi=300
)

print("Generated output/speedup.png")
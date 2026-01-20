import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("output/benchmark.csv")

latest = (
    df.groupby("filter")
      .tail(1)
)

plt.figure(figsize=(10,6))

x = range(len(latest))

plt.bar(
    [i-0.2 for i in x],
    latest["gpu_ms"],
    width=0.4,
    label="GPU"
)

plt.bar(
    [i+0.2 for i in x],
    latest["cpu_ms"],
    width=0.4,
    label="CPU"
)

plt.xticks(x, latest["filter"])

plt.ylabel("Time (ms)")
plt.title("CUDA vs CPU Performance")

plt.legend()
plt.tight_layout()

plt.savefig(
    "output/benchmark.png",
    dpi=300
)

print("Chart generated")
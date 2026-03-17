"""
Plot 2: Seed comparison at SimCLR epoch=50.
Shows high variance across seeds when embedding quality is poor.
Output: plots/seed_comparison_epoch50.png
"""
import json, os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def load(path):
    with open(path) as f:
        return json.load(f)

def get_accs(data):
    return [r["test_accuracy"] for r in data["results"]]

seeds = [10, 42, 123]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
x = [10, 20, 30, 40, 50]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Baseline (epoch=50)
for seed, c in zip(seeds, colors):
    data = load(f"results/baseline_seed{seed}.json")
    ax1.plot(x, get_accs(data), "o--", color=c, linewidth=2, markersize=7, label=f"Seed {seed}")
ax1.set_title("Baseline (SimCLR 50 epochs)", fontweight="bold")
ax1.set_xlabel("Cumulative Budget")
ax1.set_ylabel("Test Accuracy (%)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)

# Optimized (epoch=50)
for seed, c in zip(seeds, colors):
    data = load(f"results/optimized_seed{seed}.json")
    ax2.plot(x, get_accs(data), "s--", color=c, linewidth=2, markersize=7, label=f"Seed {seed}")
ax2.set_title("Early Optimized (SimCLR 50 epochs)", fontweight="bold")
ax2.set_xlabel("Cumulative Budget")
ax2.set_ylabel("Test Accuracy (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x)

fig.suptitle("High Variance Across Seeds at Low SimCLR Epochs (50)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/seed_comparison_epoch50.png", dpi=200, bbox_inches="tight")
print("Saved: plots/seed_comparison_epoch50.png")
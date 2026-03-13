"""
Plot 2: Class Distribution of selected samples — Baseline vs Optimized.
Shows whether selection achieves good class balance.

Output: plots/2_class_distribution.png
"""
import json, os
import numpy as np
import torchvision
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

with open("results/baseline.json") as f:
    base = json.load(f)
with open("results/optimized.json") as f:
    opt = json.load(f)

# Get true labels from CIFAR-10
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False)
all_labels = np.array(dataset.targets)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, data, title, color in [
    (axes[0], base, "Baseline", "#2196F3"),
    (axes[1], opt,  "Optimized", "#F44336"),
]:
    selected = np.array(data["labeled_indices"])
    sel_labels = all_labels[selected]
    counts = np.bincount(sel_labels, minlength=10)

    bars = ax.bar(range(10), counts, color=color, alpha=0.8, edgecolor="white")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) + 2)

    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Total Variation distance from uniform
    uniform = np.ones(10) / 10
    tv = 0.5 * np.sum(np.abs(counts / counts.sum() - uniform))
    ax.text(0.98, 0.95, f"TV dist = {tv:.3f}\n(lower = more balanced)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

total = len(base["labeled_indices"])
fig.suptitle(f"Class Distribution of Selected Samples ({total} total each)",
             fontsize=14, fontweight="bold", y=1)
plt.tight_layout()
plt.savefig("plots/2_class_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: plots/2_class_distribution.png")
plt.show()
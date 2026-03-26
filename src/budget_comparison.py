"""
Plot 3: Impact of different budget sizes per round.
Output: plots/budget_comparison.png
"""
import json, os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def load(path):
    with open(path) as f:
        return json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))

configs = [
    ("results/optimized_seed42.json", "Budget=10/round", "o", "#1f77b4"),
    ("results/optimized_b20.json",    "Budget=20/round", "s", "#ff7f0e"),
    ("results/optimized_b50.json",    "Budget=50/round", "^", "#2ca02c"),
]

for path, label, marker, color in configs:
    data = load(path)
    budgets = [r["budget"] for r in data["results"]]
    accs = [r["test_accuracy"] for r in data["results"]]
    ax.plot(budgets, accs, f"{marker}-", color=color, linewidth=2.5, markersize=8, label=label)

ax.set_xlabel("Cumulative Budget (labeled examples)")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Impact of Budget Size per Round (SimCLR 50 epochs)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/budget_comparison.png", dpi=200)
print("Saved: plots/budget_comparison.png")
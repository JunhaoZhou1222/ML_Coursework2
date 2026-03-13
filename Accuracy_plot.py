"""
Plot 1: Accuracy vs Budget curve.
Reads results/baseline.json and results/optimized.json.

Output: plots/1_accuracy_vs_budget.png
"""
import json, os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

with open("results/baseline.json") as f:
    base = json.load(f)
with open("results/optimized.json") as f:
    opt = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))

b_budgets = [r["budget"] for r in base["results"]]
b_acc = [r["test_accuracy"] for r in base["results"]]
o_budgets = [r["budget"] for r in opt["results"]]
o_acc = [r["test_accuracy"] for r in opt["results"]]

ax.plot(b_budgets, b_acc, "o-", color="#2196F3", linewidth=2, markersize=8, label="Baseline")
ax.plot(o_budgets, o_acc, "s-", color="#F44336", linewidth=2, markersize=8, label="Optimized")

# Annotate each point with accuracy value
for x, y in zip(b_budgets, b_acc):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=9, color="#2196F3")
for x, y in zip(o_budgets, o_acc):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color="#F44336")

ax.set_xlabel("Cumulative Budget (labeled examples)", fontsize=13)
ax.set_ylabel("Test Accuracy (%)", fontsize=13)
ax.set_title("CIFAR-10: Accuracy vs Budget", fontsize=14, fontweight="bold")
ax.legend(fontsize=12, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xticks(b_budgets)

plt.tight_layout()
plt.savefig("plots/1_accuracy_vs_budget.png", dpi=150)
print("Saved: plots/1_accuracy_vs_budget.png")
plt.show()
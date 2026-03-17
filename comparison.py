import json
import matplotlib.pyplot as plt

with open("results/original_ver_seed42.json") as f:
    orig = json.load(f)
with open("results/optimized_ver2_seed42.json") as f:
    opt = json.load(f)

budgets = [r["budget"] for r in orig["results"]]
orig_acc = [r["test_accuracy"] for r in orig["results"]]
opt_acc = [r["test_accuracy"] for r in opt["results"]]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(budgets, orig_acc, "o-", color="#1f77b4", linewidth=2.5, markersize=9, label="Original (paper params)")
ax.plot(budgets, opt_acc, "s-", color="#e74c3c", linewidth=2.5, markersize=9, label="Optimized (AutoAug+Mixup+LS)")

for x, y in zip(budgets, orig_acc):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color="#1f77b4")
for x, y in zip(budgets, opt_acc):
    ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#e74c3c")

# Improvement annotation
final_diff = opt_acc[-1] - orig_acc[-1]
ax.annotate(f"+{final_diff:.1f}%", xy=(budgets[-1], opt_acc[-1]),
            xytext=(budgets[-1]+2, opt_acc[-1]+1.5), fontsize=11, fontweight="bold", color="#e74c3c")

ax.set_xlabel("Cumulative Budget (labeled examples)", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("CIFAR-10 Low-Budget: Original vs Optimized (seed=42)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xticks(budgets)

plt.tight_layout()
plt.savefig("plots/final_acc_comparison_plot.png", dpi=200)
print("Saved")
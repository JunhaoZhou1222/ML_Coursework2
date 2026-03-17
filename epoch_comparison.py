"""
Plot 4: SimCLR 50 vs 500 epochs, and original vs optimized.
Explains why more SimCLR training matters.
Output: plots/epoch50_vs_epoch500.png
"""
import json, os
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

def load(path):
    with open(path) as f:
        return json.load(f)

def get_accs(data):
    return [r["test_accuracy"] for r in data["results"]]

baseline_50  = load("results/baseline_seed42.json")
original_500 = load("results/original_ver_seed42.json")
optimized_500 = load("results/optimized_ver2_seed42.json")

x = [10, 20, 30, 40, 50]

fig, ax = plt.subplots(figsize=(9, 5.5))

acc_50  = get_accs(baseline_50)
acc_orig = get_accs(original_500)
acc_opt  = get_accs(optimized_500)

ax.plot(x, acc_50, "o--", color="#95a5a6", linewidth=2, markersize=7,
        label=f"Baseline (50ep) — {acc_50[-1]:.1f}%")
ax.plot(x, acc_orig, "o-", color="#3498db", linewidth=2.5, markersize=8,
        label=f"Original (500ep, paper) — {acc_orig[-1]:.1f}%")
ax.plot(x, acc_opt, "s-", color="#e74c3c", linewidth=2.5, markersize=8,
        label=f"Optimized (500ep, AutoAug+Mixup+LS) — {acc_opt[-1]:.1f}%")

for accs, color, offset in [(acc_50, "#95a5a6", -12), (acc_orig, "#3498db", -12), (acc_opt, "#e74c3c", 10)]:
    ax.annotate(f"{accs[-1]:.1f}%", (50, accs[-1]),
                textcoords="offset points", xytext=(8, offset),
                fontsize=10, fontweight="bold", color=color)

ax.set_xlabel("Cumulative Budget (labeled examples)", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("SimCLR 50 vs 500 Epochs: Why More Training Matters", fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xticks(x)
plt.tight_layout()
plt.savefig("plots/epoch50_vs_epoch500.png", dpi=200)
print("Saved: plots/epoch50_vs_epoch500.png")
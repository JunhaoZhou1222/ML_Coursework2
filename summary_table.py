"""
Plot 1: Summary table of all experiments.
Output: plots/summary_table.png
"""
import json, os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 11})
os.makedirs("plots", exist_ok=True)

def load(path):
    with open(path) as f:
        return json.load(f)

all_data = [
    load("results/baseline_seed10.json"),
    load("results/baseline_seed42.json"),
    load("results/baseline_seed123.json"),
    load("results/optimized_seed10.json"),
    load("results/optimized_seed42.json"),
    load("results/optimized_seed123.json"),
    load("results/optimized_b20.json"),
    load("results/optimized_b50.json"),
    load("results/original_ver_seed42.json"),
    load("results/optimized_ver2_seed42.json"),
]

headers = ["Experiment", "SimCLR\nEpochs", "Cls\nEpochs", "Max\nClusters", "Seed",
           "R1", "R2", "R3", "R4", "R5", "Final"]

rows = []
for data in all_data:
    s = data["settings"]
    accs = [r["test_accuracy"] for r in data["results"]]
    seed = s.get("seed", s.get("seeds_run", ["?"])[0] if "seeds_run" in s else "?")
    row = [data["name"], s["simclr_epochs"], s["classifier_epochs"], s["max_clusters"], seed]
    for a in accs:
        row.append(f"{a:.1f}")
    row.append(f"{accs[-1]:.1f}")
    rows.append(row)

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis("off")

table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.4)

for j in range(len(headers)):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")

for i in range(1, len(rows) + 1):
    color = "#f0f4f8" if i % 2 == 0 else "white"
    for j in range(len(headers)):
        table[i, j].set_facecolor(color)

best_final = max(float(r[-1]) for r in rows)
for i, r in enumerate(rows):
    if float(r[-1]) == best_final:
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor("#d4edda")

ax.set_title("Summary of All Experiments", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("plots/summary_table.png", dpi=200, bbox_inches="tight")
print("Saved: plots/summary_table.png")
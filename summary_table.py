"""
Generate summary CSV from all experiment results.
Output: results/summary_table.csv
"""
import json, csv, os

def load(path):
    with open(path) as f:
        return json.load(f)

files = [
    "results/baseline_seed10.json",
    "results/baseline_seed42.json",
    "results/baseline_seed123.json",
    "results/optimized_seed10.json",
    "results/optimized_seed42.json",
    "results/optimized_seed123.json",
    "results/optimized_b20.json",
    "results/optimized_b50.json",
    "results/original_ver_seed42.json",
    "results/optimized_ver2_seed42.json",
]

headers = ["name", "simclr_epochs", "classifier_epochs", "max_clusters", "seed",
           "round1_acc", "round2_acc", "round3_acc", "round4_acc", "round5_acc", "final_acc"]

rows = []
for path in files:
    data = load(path)
    s = data["settings"]
    accs = [r["test_accuracy"] for r in data["results"]]
    seed = s.get("seed", s.get("seeds_run", [""])[0] if "seeds_run" in s else "")
    row = [data["name"], s["simclr_epochs"], s["classifier_epochs"], s["max_clusters"], seed]
    row += accs
    row.append(accs[-1])
    rows.append(row)

os.makedirs("results", exist_ok=True)
with open("results/summary_table.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print("Saved: results/summary_table.csv")
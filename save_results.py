"""
Run the current TypiClust pipeline and save results to a JSON file.

Usage:
    python save_results.py baseline         # Save as results/baseline.json
    python save_results.py optimized        # Save as results/optimized.json
    python save_results.py whatever_name    # Save as results/whatever_name.json
    python save_results.py optimized_b20 --epochs 200 --budget 20   # budget=20
    Add --full for full 500 epoch SimCLR (default: 20 epochs quick test)
    python save_results.py baseline --full
"""
import os, sys, json
import numpy as np

from typiclust import run_typiclust_rp
from typiclust.config import DEVICE, SEED, set_seed
from typiclust.embeddings import extract_embeddings

# ── Parse args ──
args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]

if len(args) < 1:
    print("Usage: python save_results.py <name> [--full] [--epochs N]")
    print("  e.g. python save_results.py baseline")
    print("       python save_results.py baseline --epochs 200")
    print("       python save_results.py optimized --full")
    sys.exit(1)

name = args[0]
FULL = "--full" in flags

# Defaults
EPOCHS = 500 if FULL else 20
CLS_EPOCHS = 100 if FULL else 50
MAX_K = 500 if FULL else 50
BUDGET = 10
ROUNDS = 5

# Parse --key value pairs
def parse_flag(flag_name, default):
    for f in flags:
        if f == flag_name:
            idx = sys.argv.index(f)
            if idx + 1 < len(sys.argv):
                return int(sys.argv[idx + 1])
    return default

EPOCHS = parse_flag("--epochs", EPOCHS)
BUDGET = parse_flag("--budget", BUDGET)
ROUNDS = parse_flag("--rounds", ROUNDS)

os.makedirs("results", exist_ok=True)

print(f"Name: {name}")
print(f"SimCLR Epochs: {EPOCHS}")
print(f"Classifier Epochs: {CLS_EPOCHS}")
print(f"Budget per round:  {BUDGET}")
print(f"Rounds:            {ROUNDS}")
print(f"Device: {DEVICE}\n")

# ── Run pipeline ──
set_seed(SEED)
results, labeled = run_typiclust_rp(
    dataset_root="./data",
    budget_per_round=BUDGET,
    num_rounds=ROUNDS,
    simclr_epochs=EPOCHS,
    classifier_epochs=CLS_EPOCHS,
    max_clusters=MAX_K,
    K_typicality=20,
)

# ── Save ──
output = {
    "name": name,
    "settings": {
        "simclr_epochs": EPOCHS,
        "classifier_epochs": CLS_EPOCHS,
        "max_clusters": MAX_K,
        "budget_per_round": BUDGET,
        "num_rounds": ROUNDS,
    },
    "results": results,
    "labeled_indices": labeled,
}

path = f"results/{name}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to: {path}")
print("Now you can run visualization scripts after saving both versions.")
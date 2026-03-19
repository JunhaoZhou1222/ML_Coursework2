"""
Demo:5 epoches AL,  Budget=10。
"""
from typiclust import run_typiclust_rp
from typiclust.config import DEVICE


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    results, final_labeled = run_typiclust_rp(
        dataset_root="./data",
        budget_per_round=10,
        num_rounds=5,
        simclr_epochs=500,       
        classifier_epochs=300, #200 in paper, 300 for better performance
        max_clusters=500,
        K_typicality=20,
    )

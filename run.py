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
        simclr_epochs=500,       # 快速测试；完整复现用 500
        classifier_epochs=200,  # 完整复现用 200
        max_clusters=500,
        K_typicality=20,
    )

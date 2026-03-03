"""
TPC_RP 入口脚本。

Demo：5 轮主动学习，每轮预算 B=10。
快速测试可将 simclr_epochs=10, classifier_epochs=20；
完整复现可设为 simclr_epochs=500, classifier_epochs=200。
"""
from typiclust import run_typiclust_rp

if __name__ == "__main__":
    results, final_labeled = run_typiclust_rp(
        dataset_root="./data",
        budget_per_round=10,
        num_rounds=5,
        simclr_epochs=10,
        classifier_epochs=20,
        max_clusters=50,
        K_typicality=20,
    )

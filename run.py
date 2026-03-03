"""
TPC_RP 入口脚本。

Demo：5 轮主动学习，每轮预算 B=10。
论文附录 F.1：SimCLR 500 epoch、分类器 200 epoch；完整复现请用下方注释参数。
快速测试建议至少 simclr_epochs=50, classifier_epochs=100（10 epoch 的 SimCLR 几乎随机，结果无意义）。
"""
from typiclust import run_typiclust_rp
from typiclust.config import DEVICE

print(f"Using device: {DEVICE}")

if __name__ == "__main__":
    results, final_labeled = run_typiclust_rp(
        dataset_root="./data",
        budget_per_round=10,
        num_rounds=5,
        simclr_epochs=50,       # 快速测试；完整复现用 500
        classifier_epochs=100,  # 完整复现用 200
        max_clusters=50,
        K_typicality=20,
    )

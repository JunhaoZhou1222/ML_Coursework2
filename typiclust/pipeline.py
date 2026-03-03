"""
TPC_RP 完整流程：SimCLR → 嵌入 → 聚类 + 典型性选择 → 分类器训练与评估。
"""
import numpy as np

from .embeddings import extract_embeddings
from .selection import typiclust_rp_select
from .train import train_simclr, train_classifier


def run_typiclust_rp(
    dataset_root: str = "./data",
    budget_per_round: int = 10,
    num_rounds: int = 5,
    simclr_epochs: int = 50,
    classifier_epochs: int = 100,
    max_clusters: int = 500,
    K_typicality: int = 20,
):
    """
    CIFAR-10 上的完整 TPC_RP 流程。

    Round 0 = 初始池选择（L0 为空），之后每轮增加 B 个样本。
    """
    print("=" * 60)
    print("  TPC_RP: TypiClust with SimCLR + K-means on CIFAR-10")
    print("=" * 60)

    encoder = train_simclr(
        dataset_root=dataset_root,
        epochs=simclr_epochs,
        batch_size=256,
        lr=0.5,
        temperature=0.5,
        projection_dim=128,
    )
    embeddings, true_labels = extract_embeddings(
        encoder,
        dataset_root=dataset_root,
        batch_size=512,
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    labeled_indices = []
    results = []

    for round_idx in range(num_rounds):
        print(f"\n{'─' * 60}")
        print(f"  Active Learning Round {round_idx + 1}/{num_rounds}")
        print(f"  Current labeled set size: {len(labeled_indices)}")
        print(f"{'─' * 60}")

        new_queries = typiclust_rp_select(
            embeddings=embeddings,
            budget=budget_per_round,
            max_clusters=max_clusters,
            existing_labeled_indices=labeled_indices,
            K_typicality=K_typicality,
        )

        labeled_indices = labeled_indices + new_queries
        total_budget = len(labeled_indices)
        print(f"  Newly queried  : {len(new_queries)}  |  Total labeled: {total_budget}")

        queried_labels = true_labels[np.array(labeled_indices)]
        unique, counts = np.unique(queried_labels, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"  Class distribution: {class_dist}")

        print(f"  Training classifier on {total_budget} labeled examples ...")
        acc = train_classifier(
            labeled_indices=labeled_indices,
            dataset_root=dataset_root,
            epochs=classifier_epochs,
            batch_size=min(64, total_budget),
            lr=0.025,
        )
        results.append({
            "round": round_idx + 1,
            "budget": total_budget,
            "test_accuracy": acc,
        })
        print(f"  ✓ Test Accuracy: {acc:.2f}%")

    print("\n" + "=" * 60)
    print("  TPC_RP Results Summary")
    print("=" * 60)
    print(f"  {'Round':>5}  {'Budget':>8}  {'Test Acc (%)':>12}")
    print(f"  {'─' * 5}  {'─' * 8}  {'─' * 12}")
    for r in results:
        print(f"  {r['round']:>5}  {r['budget']:>8}  {r['test_accuracy']:>12.2f}")

    return results, labeled_indices

"""
TPC_RP 核心：基于 K-means + 典型性的主动学习样本选择。
"""
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .config import SEED
from .typicality import compute_typicality


def typiclust_rp_select(
    embeddings: np.ndarray,
    budget: int,
    max_clusters: int = 500,
    existing_labeled_indices: Optional[List[int]] = None,
    K_typicality: int = 20,
) -> List[int]:
    """
    论文 Algorithm 1：TypiClust 初始池选择（TPC_RP 变体）。

    步骤：
    1. K-means 聚类为 min(|L_{i-1}| + B, max_clusters) 类
    2. 标记已被已有标签覆盖的簇
    3. 从 B 个最大未覆盖簇中各选一个最典型样本

    Parameters
    ----------
    embeddings               : np.ndarray (N, D)
    budget                   : int  B，本轮查询数量
    max_clusters             : int  簇数上界（论文 CIFAR 用 500）
    existing_labeled_indices : list[int] | None  已标注索引 L_{i-1}
    K_typicality             : int  典型性计算的 K

    Returns
    -------
    query_indices : list[int]  选中的 B 个样本索引
    """
    if existing_labeled_indices is None:
        existing_labeled_indices = []

    n_existing = len(existing_labeled_indices)
    n_clusters = min(n_existing + budget, max_clusters)
    n_clusters = max(n_clusters, budget)

    print(f"\n=== Step 2: K-means clustering into {n_clusters} clusters ===")
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    else:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            batch_size=1024,
            n_init=3,
        )
    cluster_ids = km.fit_predict(embeddings)

    print(f"\n=== Step 3: Computing typicality for all {len(embeddings)} points ===")
    typicality = compute_typicality(embeddings, K=K_typicality)

    covered_clusters = set()
    for idx in existing_labeled_indices:
        covered_clusters.add(cluster_ids[idx])

    cluster_map = {}
    for i in range(len(embeddings)):
        cid = cluster_ids[i]
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append((typicality[i], i))

    uncovered = [
        (cid, pts)
        for cid, pts in cluster_map.items()
        if cid not in covered_clusters
    ]
    uncovered.sort(key=lambda x: len(x[1]), reverse=True)

    print(f"  Covered clusters   : {len(covered_clusters)}")
    print(f"  Uncovered clusters : {len(uncovered)}")

    query_indices = []
    MIN_CLUSTER_SIZE = 5
    existing_set = set(existing_labeled_indices)

    for cid, pts in uncovered:
        if len(query_indices) >= budget:
            break
        valid_pts = [(t, idx) for t, idx in pts if idx not in existing_set]
        if len(valid_pts) < MIN_CLUSTER_SIZE:
            continue
        best_idx = max(valid_pts, key=lambda x: x[0])[1]
        query_indices.append(best_idx)

    if len(query_indices) < budget:
        already_selected = set(query_indices) | existing_set
        remaining = sorted(
            [
                (typicality[i], i)
                for i in range(len(embeddings))
                if i not in already_selected
            ],
            reverse=True,
        )
        for _, idx in remaining:
            if len(query_indices) >= budget:
                break
            query_indices.append(idx)

    return query_indices[:budget]

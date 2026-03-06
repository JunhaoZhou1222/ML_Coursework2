"""
Typicality(x) = (1/K * sum of euclidean x1 and x2)^{-1}
 min{20, cluster_size} nearest neighbors
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(embeddings: np.ndarray, K: int = 20) -> np.ndarray:
    """
    从全部50000张图里找最近的20张
    """
    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, len(embeddings)),
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_dist = distances[:, 1:].mean(axis=1)  # calculate mean of distances[:, 1:]
    avg_dist = np.clip(avg_dist, a_min=1e-8, a_max=None)  # 防止两点完全重合
    return 1.0 / avg_dist  # low mean distance means high typicality


def compute_typicality_per_cluster(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    K: int = 20,
) -> np.ndarray:
    """
    只在自己的簇里找邻居
    """
    typicality = np.full(embeddings.shape[0], np.nan, dtype=np.float64)
    for cid in np.unique(cluster_ids):
        # 找出属于当前簇 cid 的所有样本
        mask = cluster_ids == cid
        inds = np.where(mask)[0]
        sub_emb = embeddings[inds]  # 只取出这个簇的 embedding
        # min{20, cluster_size}；max cluster_size-1 nbrs
        k_eff = min(K, len(inds) - 1)
        if k_eff < 1: #only one sample in cluster
            continue

        nbrs = NearestNeighbors(
            n_neighbors=k_eff + 1,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        nbrs.fit(sub_emb)

        distances, _ = nbrs.kneighbors(sub_emb)
        avg_dist = distances[:, 1:].mean(axis=1)
        avg_dist = np.clip(avg_dist, a_min=1e-8, a_max=None)
        typicality[inds] = 1.0 / avg_dist
    return typicality

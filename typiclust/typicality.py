"""
典型性计算（论文 Eq.4 + 附录 F.1）。
Typicality(x) = (1/K * sum_{x_i in K-NN(x)} ||x - x_i||_2)^{-1}
附录 F.1：we used min{20, cluster_size} nearest neighbors —— 按簇自适应 K。
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(embeddings: np.ndarray, K: int = 20) -> np.ndarray:
    """
    全局典型性（不按簇）：到 K 近邻平均距离的倒数。
    仅当不提供 cluster_ids 时使用；否则应使用 compute_typicality_per_cluster。
    """
    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, len(embeddings)),
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    avg_dist = distances[:, 1:].mean(axis=1)
    avg_dist = np.clip(avg_dist, a_min=1e-8, a_max=None)
    return 1.0 / avg_dist


def compute_typicality_per_cluster(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    K: int = 20,
) -> np.ndarray:
    """
    按簇计算典型性，每簇使用 K_eff = min(K, cluster_size) 个近邻（附录 F.1）。
    小簇不会出现 K 超过簇大小的情况。

    Parameters
    ----------
    embeddings  : np.ndarray (N, D)
    cluster_ids : np.ndarray (N,)  每个样本的簇 id
    K           : int  近邻数上界（论文 20）

    Returns
    -------
    typicality : np.ndarray (N,)  未参与计算的点（如单点簇）为 np.nan
    """
    typicality = np.full(embeddings.shape[0], np.nan, dtype=np.float64)
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        inds = np.where(mask)[0]
        sub_emb = embeddings[inds]
        # 附录 F.1: min{20, cluster_size}；排除自身后最多 cluster_size-1 个邻居
        k_eff = min(K, len(inds) - 1)
        if k_eff < 1:
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

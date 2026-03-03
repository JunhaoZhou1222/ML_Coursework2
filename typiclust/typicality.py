"""
典型性计算（论文 Eq.4）。
Typicality(x) = (1/K * sum_{x_i in K-NN(x)} ||x - x_i||_2)^{-1}
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_typicality(embeddings: np.ndarray, K: int = 20) -> np.ndarray:
    """
    典型性 = 到 K 近邻平均距离的倒数。

    Parameters
    ----------
    embeddings : np.ndarray  (N, D)
    K          : int        近邻数（论文默认 20）

    Returns
    -------
    typicality : np.ndarray  (N,)
    """
    # +1 因为自身总是最近邻
    nbrs = NearestNeighbors(
        n_neighbors=K + 1,
        algorithm="auto",
        metric="euclidean",
        n_jobs=-1,
    )
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    avg_dist = distances[:, 1:].mean(axis=1)
    avg_dist = np.clip(avg_dist, a_min=1e-8, a_max=None)
    typicality = 1.0 / avg_dist
    return typicality

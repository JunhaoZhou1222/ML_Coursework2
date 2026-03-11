"""
TypiClust sample selection (Algorithm 1 from the paper):
  1. K-means clustering on embeddings
  2. Mark clusters that already contain a labeled sample as "covered"
  3. From the B largest uncovered clusters, pick the most typical sample each
"""
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .config import SEED
from .typicality import compute_typicality_per_cluster


def typiclust_rp_select(
    embeddings,
    budget,  
    max_clusters = 500,
    existing_labeled_indices = None,
    K_typicality = 20,
):
    # First round, no labels
    if existing_labeled_indices is None:
        existing_labeled_indices = []

    # K-means clustetring
    n_existing = len(existing_labeled_indices)
    n_clusters = min(n_existing + budget, max_clusters) #clusters = existedlabel + B
    #n_clusters = max(n_clusters, budget)

    print(f"\n=== Step 2: K-means clustering into {n_clusters} clusters ===")
    #In paper, it used KMeans when K ≤ 50 and MiniBatchKMeans otherwise.
    if n_clusters <= 50:
        km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    else:
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=SEED,
            batch_size=1024,
            n_init=3,
        )
    cluster_ids = km.fit_predict(embeddings) #cluster_ids = [0, 2, 0, 1, 2, ...]

    # caclulate typicality
    print(f"\n=== Step 3: Computing typicality (per-cluster K_eff=min({K_typicality}, cluster_size)) ===")
    typicality = compute_typicality_per_cluster(
        embeddings, cluster_ids, K=K_typicality
    )
    # clusters with labeled sample = covered clusters
    covered_clusters = {cluster_ids[idx] for idx in existing_labeled_indices}
    
    cluster_map = {}
    for i in range(len(embeddings)):
        if np.isnan(typicality[i]):
            continue
        cid = cluster_ids[i]
        if cid not in cluster_map:
            cluster_map[cid] = []
        cluster_map[cid].append((typicality[i], i))

    uncovered = [
        (cid, pts)
        for cid, pts in cluster_map.items()
        if cid not in covered_clusters # clusters with unlabeled sample
    ]
    uncovered.sort(key=lambda x: len(x[1]), reverse=True) # rank from high density to low

    print(f"  Covered clusters   : {len(covered_clusters)}")
    print(f"  Uncovered clusters : {len(uncovered)}")

    query_indices = []
    MIN_CLUSTER_SIZE = 5
    existing_set = set(existing_labeled_indices)

    for cid, pts in uncovered:
        if len(query_indices) >= budget:
            break
        valid_pts = [(t, idx) for t, idx in pts if idx not in existing_set]
        if len(valid_pts) < MIN_CLUSTER_SIZE: # the number of sample is not enough
            continue
        best_idx = max(valid_pts, key=lambda x: x[0])[1]
        query_indices.append(best_idx)

    if len(query_indices) < budget:
        print(f"  Warning: Only {len(query_indices)}/{budget} selected (not enough large uncovered clusters).")

    return query_indices[:budget]

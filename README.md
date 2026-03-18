# ML Coursework 2 — What this repo does (and what we optimized)

This repository implements a **TypiClust-style active learning (AL)** pipeline on **CIFAR-10**:

- **Representation learning**: train a SimCLR encoder and extract L2-normalized embeddings for the training set.
- **Sample selection (AL)**: run K-means in embedding space, find **uncovered clusters**, and query the **most typical** point per large uncovered cluster.
- **Supervised training**: train a classifier on the growing labeled set and report **test accuracy** after each AL round.

## Improvements and Implementation

To improve the model’s robustness under low-budget conditions, this experiment introduced five key training optimizations based on the original algorithm:

- **Mixup**: Images within the same data batch are randomly paired, and two different images and their labels are fused at the pixel level. According to H. Zhang et al. (2017) regarding Mixup: Beyond Empirical Risk Minimization, this technique enables the model to learn smooth transition characteristics between different categories, thereby avoiding reliance on simple binary memory. As a result, overfitting is significantly reduced, and the model’s generalisation ability and robustness to noise are improved.
- **AutoAugment**: With very few samples, models tend to overfit to background noise instead of learning true semantic features. Following Cubuk et al. (2019) regarding AutoAugment research, we applied AutoAugment, which leverages reinforcement learning to discover optimal data augmentation strategies. By introducing strong composite transformations (e.g., shearing, rotation, and hue distortion), it acts as a powerful regularizer that reduces the generalization gap between the training and test sets.
- **Label Smoothing**: To prevent the model from overfitting, we used a technique called label smoothing (Szegedy et al., 2016). This method changes the usual "one-hot" hard labels into softer probability distributions by adding a small amount of noise. This helps control the extreme growth of the log-odds ratio. As a result, the decision boundary becomes smoother and more stable. It also improves the model’s ability to handle unusual data points, even when there are very few training examples. This leads to better prediction accuracy and overall performance.
- **Decrease Temperature**: In the context of SimCLR's contrastive loss, reducing the temperature parameter from 0.5 to 0.1 makes the model more sensitive to hard negatives. This creates a stricter, more highly separated feature space, resulting in embeddings that are more stable, precise, and consistent for downstream classification.
- **More Classifier Epochs**: These additional 100 rounds of classifier training enable the model to fine-tune its decision boundaries more precisely, thereby perfectly mapping the high-quality features extracted from the lower layers to the correct class labels.

## Results: original vs optimized (from saved outputs)

The repo contains both an “original” run and an “optimized” run (same seed, same budget schedule).

- **Original**: `results/original_ver_seed42.json`  
  - Final accuracy (round 5, budget 50): **25.02%**
- **Optimized (v2)**: `results/optimized_ver2_seed42.json`  
  - Final accuracy (round 5, budget 50): **28.76%**

That is a **+3.74 pp** improvement at the final AL round for seed 42.

## What gets produced (outputs)

- **Raw per-run outputs (JSON)**: `results/*.json`  
  Each file includes `settings`, per-round `{round, budget, test_accuracy}`, and `labeled_indices`.
- **CSV summary (already in repo)**: `results/summary_table.csv`  
  One row per experiment with round accuracies and final accuracy.
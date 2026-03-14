import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置画图全局参数，使其看起来更像学术论文配图
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'legend.fontsize': 10,
    'grid.alpha': 0.5
})

# 辅助函数：读取 JSON 文件中的准确率数据
def get_accuracies(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            budgets = [step['budget'] for step in data['results']]
            accs = [step['test_accuracy'] for step in data['results']]
            return budgets, accs
    except FileNotFoundError:
        print(f"Warning: File {filepath} not found.")
        return [], []

# 我们要分析的种子
seeds = [10, 42, 123]

# =========================================================
# 图 1: Baseline vs Optimized (带误差棒的平均曲线)
# =========================================================
def plot_baseline_vs_optimized():
    plt.figure(figsize=(8, 5))
    
    # 存储所有种子的数据
    base_accs_all = []
    opt_accs_all = []
    
    for seed in seeds:
        _, b_acc = get_accuracies(f"results/baseline_seed{seed}.json")
        _, o_acc = get_accuracies(f"results/optimized_seed{seed}.json")
        if b_acc and o_acc:
            base_accs_all.append(b_acc)
            opt_accs_all.append(o_acc)
            
    # X轴刻度 (因为 budget_per_round=10)
    x = [10, 20, 30, 40, 50]
    
    if base_accs_all and opt_accs_all:
        base_mean = np.mean(base_accs_all, axis=0)
        base_std = np.std(base_accs_all, axis=0)
        
        opt_mean = np.mean(opt_accs_all, axis=0)
        opt_std = np.std(opt_accs_all, axis=0)
        
        # 绘制 Baseline
        plt.plot(x, base_mean, label='Baseline (Mean)', marker='o', color='#1f77b4')
        plt.fill_between(x, base_mean - base_std, base_mean + base_std, color='#1f77b4', alpha=0.2)
        
        # 绘制 Optimized
        plt.plot(x, opt_mean, label='Optimized (Mean)', marker='s', color='#ff7f0e')
        plt.fill_between(x, opt_mean - opt_std, opt_mean + opt_std, color='#ff7f0e', alpha=0.2)

    plt.title('Baseline vs Optimized (Budget=10 per round)')
    plt.xlabel('Cumulative Labeled Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig1_Baseline_vs_Optimized.png', dpi=300)
    print("Saved: Fig1_Baseline_vs_Optimized.png")

# =========================================================
# 图 2: 不同随机种子的波动对比 (以 Optimized 为例)
# =========================================================
def plot_seed_variance():
    plt.figure(figsize=(8, 5))
    colors = ['#2ca02c', '#d62728', '#9467bd']
    markers = ['^', 'v', 'D']
    x = [10, 20, 30, 40, 50]
    
    for i, seed in enumerate(seeds):
        _, acc = get_accuracies(f"results/optimized_seed{seed}.json")
        if acc:
            plt.plot(x, acc, label=f'results/Optimized (Seed {seed})', 
                     marker=markers[i], color=colors[i], linestyle='--')

    plt.title('High Variance Across Seeds in Ultra-low Budget (Optimized)')
    plt.xlabel('Cumulative Labeled Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(x)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig2_Seed_Variance.png', dpi=300)
    print("Saved: Fig2_Seed_Variance.png")

# =========================================================
# 图 3: 不同 Budget 策略的对比 (b10 vs b20 vs b50)
# =========================================================
def plot_budget_comparison():
    plt.figure(figsize=(8, 5))
    
    # 读取三种不同 budget 的文件
    files_labels = [
        ("results/optimized.json", "Budget=10 per round", "o", "#1f77b4"),
        ("results/optimized_b20.json", "Budget=20 per round", "s", "#ff7f0e"),
        ("results/optimized_b50.json", "Budget=50 per round", "^", "#2ca02c")
    ]
    
    for filename, label, marker, color in files_labels:
        budgets, accs = get_accuracies(filename)
        if budgets and accs:
            # 画线和散点
            plt.plot(budgets, accs, label=label, marker=marker, color=color)

    plt.title('Impact of Query Budget Size on Accuracy')
    plt.xlabel('Cumulative Labeled Samples')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Fig3_Budget_Comparison.png', dpi=300)
    print("Saved: Fig3_Budget_Comparison.png")

if __name__ == '__main__':
    # 假设你的 json 文件都在同一个目录下（如果存在 results/ 文件夹里，请将上面的文件名加上 'results/' 前缀）
    plot_baseline_vs_optimized()
    plot_seed_variance()
    plot_budget_comparison()
    print("All plots generated successfully!")
"""
汇总 results 目录下所有 JSON 实验结果，并导出为 CSV 表格。
"""
import json
import csv
from pathlib import Path


def load_result(path: Path) -> dict | None:
    """加载单个 JSON 结果文件。"""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"警告: 无法读取 {path}: {e}")
        return None


def summarize_to_row(data: dict) -> dict:
    """将单次实验的 JSON 转为 CSV 的一行（字典）。"""
    name = data.get("name", "")
    settings = data.get("settings", {})
    results = data.get("results", [])
    labeled = data.get("labeled_indices", [])

    row = {
        "name": name,
        "simclr_epochs": settings.get("simclr_epochs", ""),
        "classifier_epochs": settings.get("classifier_epochs", ""),
        "max_clusters": settings.get("max_clusters", ""),
        "budget_per_round": settings.get("budget_per_round", ""),
        "num_rounds": settings.get("num_rounds", ""),
        "seed": settings.get("seed") or (settings.get("seeds_run", [None])[0] if settings.get("seeds_run") else ""),
        "num_labeled": len(labeled),
    }

    # 每轮 test_accuracy
    for r in results:
        round_num = r.get("round")
        acc = r.get("test_accuracy")
        if round_num is not None:
            row[f"round_{round_num}_accuracy"] = acc if acc is not None else ""

    # 汇总：最后一轮精度、平均精度
    accs = [r.get("test_accuracy") for r in results if r.get("test_accuracy") is not None]
    row["final_accuracy"] = round(accs[-1], 2) if accs else ""
    row["mean_accuracy"] = round(sum(accs) / len(accs), 2) if accs else ""

    return row


def get_all_round_columns(rows: list[dict]) -> list[str]:
    """根据所有行中出现的 round_X_accuracy 列，得到完整列名列表（按 round 排序）。"""
    round_cols = set()
    for r in rows:
        round_cols.update(k for k in r if k.startswith("round_") and k.endswith("_accuracy"))
    return sorted(round_cols, key=lambda x: int(x.replace("round_", "").replace("_accuracy", "")))


def main():
    results_dir = Path(__file__).parent / "results"
    output_csv = Path(__file__).parent / "results_summary.csv"

    if not results_dir.is_dir():
        print(f"错误: 未找到目录 {results_dir}")
        return

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"未在 {results_dir} 中找到 JSON 文件")
        return

    rows = []
    for path in json_files:
        data = load_result(path)
        if data is None:
            continue
        rows.append(summarize_to_row(data))

    if not rows:
        print("没有可汇总的结果")
        return

    # 固定列顺序：基本信息 -> 各轮精度 -> 汇总
    round_cols = get_all_round_columns(rows)
    base_cols = [
        "name", "simclr_epochs", "classifier_epochs", "max_clusters",
        "budget_per_round", "num_rounds", "seed", "num_labeled",
    ]
    summary_cols = ["final_accuracy", "mean_accuracy"]
    fieldnames = base_cols + round_cols + summary_cols

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"已汇总 {len(rows)} 个实验，输出到: {output_csv}")


if __name__ == "__main__":
    main()

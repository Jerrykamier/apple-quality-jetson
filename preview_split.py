"""划分预览：只从原 train 中按同源组留出约 20%，不复制或修改图片。

运行 python preview_split.py，结果写入 data/splits 下的 JSON 文件。
只使用 Python 标准库，不需要安装依赖。原 test 不参与本脚本的计算。
"""
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SEED = 42
VAL_RATIO = 0.20
CANDIDATES = 1000
LABELS = ("freshapples", "rottenapples")
# 只去掉已经核实的增强前缀，后面的完整原图文件名作为来源标识。
PREFIXES = tuple(f"rotated_by_{angle}_" for angle in (15, 30, 45, 60, 75)) + (
    "vertical_flip_", "translation_", "saltandpepper_",
)


def identify(name):
    """返回（原图文件名，图片类型）。类型只根据命名前缀识别。"""
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):], prefix.rstrip("_")
    return name, "original"


def make_plan(files):
    """files 是（类别，文件名）的列表；本函数只计算名单，不操作图片。"""
    rows = []
    # 排序让文件遍历顺序固定，配合随机种子可以复现同一份结果。
    for label, name in sorted(files):
        source_name, kind = identify(name)
        rows.append({
            "source": f"train/{label}/{name}", "label": label,
            "group_id": f"{label}/{source_name}", "image_type": kind,
        })
    rng = random.Random(SEED)
    validation_groups = set()
    class_summary = {}
    type_summary = {}
    for label in LABELS:
        members = [row for row in rows if row["label"] == label]
        if not members:
            raise ValueError(f"No files for {label}")
        groups = defaultdict(list)
        for row in members:
            groups[row["group_id"]].append(row)
        group_ids = sorted(groups)
        if len(group_ids) < 2:
            raise ValueError(f"At least two source groups needed for {label}")
        # 两类分别抽取约 20% 的组。组大小不同，所以图片比例未必恰好 20%。
        number = max(1, min(len(group_ids) - 1, round(len(group_ids) * VAL_RATIO)))
        totals = Counter(row["image_type"] for row in members)
        group_counts = {key: Counter(row["image_type"] for row in group)
                        for key, group in groups.items()}
        target = round(len(members) * VAL_RATIO)
        best_score, best_groups = float("inf"), None
        # 生成固定数量的随机候选，只比较图片数量和类型比例，不使用模型成绩。
        # 选择最接近目标的候选；这是让随机划分尽量平衡，而不是挑测试成绩。
        for _ in range(CANDIDATES):
            chosen = rng.sample(group_ids, number)
            counts = Counter()
            for key in chosen:
                counts.update(group_counts[key])
            size_error = abs(sum(counts.values()) - target) / max(target, 1)
            type_error = sum(abs(counts[kind] - count * VAL_RATIO)
                             / max(count * VAL_RATIO, 1)
                             for kind, count in totals.items()) / len(totals)
            score = 4 * size_error + type_error
            if score < best_score:
                best_score, best_groups = score, set(chosen)
        validation_groups.update(best_groups)
        val_counts = Counter(row["image_type"] for row in members
                             if row["group_id"] in best_groups)
        val_files = sum(val_counts.values())
        class_summary[label] = {
            "original_train_files": len(members), "train_files": len(members) - val_files,
            "val_files": val_files, "val_percent": round(val_files / len(members) * 100, 2),
            "train_groups": len(groups) - len(best_groups), "val_groups": len(best_groups),
        }
        type_summary[label] = {
            kind: {"source_files": count, "train_files": count - val_counts[kind],
                   "val_files": val_counts[kind],
                   "val_percent": round(val_counts[kind] / count * 100, 2)}
            for kind, count in sorted(totals.items())
        }
    for row in rows:
        row["assigned_split"] = "val" if row["group_id"] in validation_groups else "train"
    train_groups = {row["group_id"] for row in rows if row["assigned_split"] == "train"}
    val_groups = {row["group_id"] for row in rows if row["assigned_split"] == "val"}
    # 验收已确认的要求：名单无重复、没有漏项、同源组不跨 train/val。
    assert len({row["source"] for row in rows}) == len(rows) == len(files)
    assert not train_groups.intersection(val_groups)
    return {
        "seed": SEED, "target_val_ratio": VAL_RATIO, "random_candidates_per_class": CANDIDATES,
        "scope": "Original train only. Original test is not read or included. No images modified.",
        "group_rule": "Class plus source filename after removing one recognized augmentation prefix.",
        "class_summary": class_summary, "type_summary": type_summary,
        "checks": {"all_input_files_assigned_once": True, "train_val_group_overlap": 0},
        "assignments": rows,
    }


def main():
    project = Path(__file__).resolve().parent
    root = project / "data/raw/apples/train"
    files = [(label, path.name) for label in LABELS
             for path in (root / label).iterdir() if path.is_file()]
    report = make_plan(files)
    directory = project / "data/splits"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"apple-split-preview-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report["class_summary"], ensure_ascii=False, indent=2))
    print("TYPE_COUNTS:", json.dumps(report["type_summary"], ensure_ascii=False))
    print("CHECKS:", json.dumps(report["checks"]))
    print("PREVIEW_REPORT:", output)
    print("Preview only. No images copied, moved or changed; original test not read.")


if __name__ == "__main__":
    main()

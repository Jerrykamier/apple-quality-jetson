"""按已确认的预览名单复制训练/验证图片，不重新随机划分，不读取原 test。

原始图片始终保留。复制后逐文件比较 SHA-256，并保存实际数据清单。
运行：python materialize_split.py data/splits/已确认的预览文件.json
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

from preview_split import identify

LABEL_MAP = {"freshapples": "fresh", "rottenapples": "rotten"}


def sha256(path):
    """分块读取文件，比较内容指纹，不把所有图片一次装进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(project, manifest):
    # 使用用户已经看过的名单，不在复制时生成一份新的划分。
    content = manifest.read_bytes()
    plan = json.loads(content)
    rows = plan["assignments"]
    source_root = project / "data/raw/apples"
    output = project / "data/apple-quality"
    jobs = []
    sources = set()
    group_splits = defaultdict(set)
    expected_counts = Counter()
    for row in rows:
        parts = PurePosixPath(row["source"]).parts
        if (len(parts) != 3 or parts[0] != "train" or parts[1] not in LABEL_MAP
                or parts[2] in {"", ".", ".."} or "\\" in parts[2]):
            raise ValueError(f"Unexpected source path: {row['source']}")
        split = row["assigned_split"]
        if split not in {"train", "val"} or row["label"] != parts[1]:
            raise ValueError("Invalid label or assigned split in preview")
        # 重新核对组编号，确保复制的是预览所规定的整组图片。
        base, kind = identify(parts[2])
        group_id = f"{parts[1]}/{base}"
        if row["group_id"] != group_id or row["image_type"] != kind:
            raise ValueError("Group or image type differs from preview rule")
        group_splits[group_id].add(split)
        if row["source"] in sources:
            raise ValueError("Duplicate source in preview")
        sources.add(row["source"])
        label = LABEL_MAP[parts[1]]
        relative = Path(split) / label / parts[2]
        jobs.append((source_root.joinpath(*parts), output / relative, row, relative))
        expected_counts[f"{split}/{label}"] += 1
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Source group crosses train/val")
    # 只枚举原 train：原 test 不参与覆盖检查或复制。
    actual_sources = {f"train/{label}/{path.name}" for label in LABEL_MAP
                      for path in (source_root / "train" / label).iterdir() if path.is_file()}
    if not rows or actual_sources != sources:
        raise ValueError("Original train files do not match the confirmed preview")
    for label in LABEL_MAP:
        for split in ("train", "val"):
            expected = plan["class_summary"][label][f"{split}_files"]
            if expected_counts[f"{split}/{LABEL_MAP[label]}"] != expected:
                raise ValueError("Counts differ from preview summary")

    verified = []
    copied = reused = 0
    for number, (source, target, row, relative) in enumerate(jobs, 1):
        original_hash = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            # 重复执行可复用相同内容，但不会覆盖不同内容的已有图片。
            if not target.is_file() or sha256(target) != original_hash:
                raise RuntimeError(f"Existing file differs; preserved: {target}")
            reused += 1
        else:
            temporary = target.with_name(target.name + ".copy-part")
            shutil.copyfile(source, temporary)
            if sha256(temporary) != original_hash:
                raise RuntimeError(f"Copy verification failed: {source}")
            temporary.replace(target)
            copied += 1
        verified.append({**row, "destination": relative.as_posix(), "sha256": original_hash})
        if number % 500 == 0:
            print(f"Copied and verified: {number}/{len(jobs)}", flush=True)
    # 检查实际目录里没有遗漏或额外文件。
    expected_paths = {item[3].as_posix() for item in jobs}
    actual_paths = {path.relative_to(output).as_posix() for split in ("train", "val")
                    for path in (output / split).rglob("*") if path.is_file()}
    if actual_paths != expected_paths:
        raise RuntimeError("Output file list differs from preview")
    report = {
        "source_preview": str(manifest),
        "source_preview_sha256": hashlib.sha256(content).hexdigest(),
        "dataset_root": str(output), "counts": dict(sorted(expected_counts.items())),
        "copied": copied, "reused": reused, "verified_files": len(verified),
        "train_val_group_overlap": 0, "test_included": False, "assignments": verified,
    }
    record = output / f"materialized-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    with record.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("COUNTS:", json.dumps(report["counts"]))
    print(f"PASS: {len(verified)} files verified, source group overlap 0.")
    print("REPORT:", record)
    print("Original images and original test unchanged. No training started.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按确认的名单生成训练/验证目录")
    parser.add_argument("manifest", type=Path, help="已确认的预览 JSON 路径")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    manifest = args.manifest if args.manifest.is_absolute() else project / args.manifest
    materialize(project, manifest)

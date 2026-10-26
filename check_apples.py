"""苹果图片基础检查：找损坏图片，以及文件内容完全相同的图片。

阅读顺序：定位文件夹 → 逐张读取 → 汇总重复 → 保存报告。
只读取原图片，不移动、不删除、不改写图片；报告另存在 data/audit 中。
本脚本不使用模型或 GPU，也不判断标签是否正确。
运行方法：在项目终端执行 python check_apples.py。
"""
# hashlib：为文件计算“内容指纹”，用于比较文件是否完全相同。
import hashlib
# io：把内存中的图片字节包装成 Pillow 能读取的对象。
import io
# json：把检查结果保存为有结构的文本文件，方便以后查看。
import json
# defaultdict：访问新键时，自动创建空列表或从 0 开始计数。
from collections import defaultdict
# datetime：给报告文件名加上时间，保留每次检查结果。
from datetime import datetime
# Path：表示文件或文件夹路径；用 / 可以连接下一层目录。
from pathlib import Path

# PIL 是已安装的 Pillow 图片库的导入名称。
from PIL import Image


def audit(project):
    """project 是项目根目录，例如 /home/nvidia/yolo-classify。"""
    # 1. 找到原始苹果数据，以及 train/test 下的正常、腐烂四个文件夹。
    root = project / "data/raw/apples"
    # 列表推导式：依次组合每个 split（分组）和 label（类别）。
    folders = [root / split / label for split in ("train", "test")
               for label in ("freshapples", "rottenapples")]
    for folder in folders:
        # 路径不正确时立刻停止，不把“没找到图片”当成“检查通过”。
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")
    # iterdir() 列出目录内容；is_file() 排除文件夹；sorted() 固定检查顺序。
    files = sorted(path for folder in folders for path in folder.iterdir() if path.is_file())
    if not files:
        raise RuntimeError("No files found in the dataset folders.")
    # 2. 准备记录容器。例：hashes[同一个指纹] = [图片路径1, 图片路径2]。
    hashes = defaultdict(list)
    # errors 保存读取失败的路径和原因；counts 记录各文件夹文件数。
    errors = []
    counts = defaultdict(int)
    for number, path in enumerate(files, 1):
        # enumerate(..., 1) 让进度从 1 开始；relative 是不含根目录的路径。
        relative = path.relative_to(root).as_posix()
        counts[path.parent.relative_to(root).as_posix()] += 1
        try:
            # 读取的是整个文件的二进制内容，不根据文件名判断重复。
            data = path.read_bytes()
            # 内容相同会得到相同 SHA-256 指纹，并归入同一组。
            # 旋转、裁剪或重新编码会改变文件内容，本方法不查这类相似图片。
            hashes[hashlib.sha256(data).hexdigest()].append(relative)
            # with 在代码块结束时自动关闭图片对象。
            # verify() 检查图片格式/完整性；load() 实际解码读取像素。
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            # verify() 之后必须重新打开图片，才能进行 load()。
            with Image.open(io.BytesIO(data)) as image:
                image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
            # 某张图片出问题，只记录错误，继续检查其他图片。
            errors.append({"path": relative, "error": f"{type(error).__name__}: {error}"})
        if number % 500 == 0:
            # % 是取余；每检查 500 个文件显示一次进度。flush=True 立即显示。
            print(f"Checked: {number}/{len(files)}", flush=True)
    # 3. 找出同一指纹下至少有两个文件的组，即完全重复组。
    duplicates = []
    for digest, paths in sorted(hashes.items()):
        if len(paths) < 2:
            # continue 跳过当前组，继续检查下一组。
            continue
        # 路径形如 train/freshapples/a.png：第 0 段是分组，第 1 段是类别。
        # {...} 是集合，会去掉重复值；集合长度 > 1 表示跨组或跨类别。
        duplicates.append({
            "sha256": digest,
            "paths": paths,
            "cross_split": len({p.split('/')[0] for p in paths}) > 1,
            "cross_label": len({p.split('/')[1] for p in paths}) > 1,
        })
    # 4. 汇总报告。重复组数不是重复图片张数，一组可能有多张图片。
    # Python 中 True 可按 1、False 可按 0 累加，所以 sum(...) 能统计组数。
    summary = {
        # 依次为：已检查文件、图片错误、完全重复组、跨分组重复、跨类别重复。
        "files_checked": len(files),
        "image_errors": len(errors),
        "exact_duplicate_groups": len(duplicates),
        "cross_split_duplicate_groups": sum(group["cross_split"] for group in duplicates),
        "cross_label_duplicate_groups": sum(group["cross_label"] for group in duplicates),
    }
    # summary 是摘要，image_errors / exact_duplicates 是具体文件明细。
    report = {
        "dataset_root": str(root), "summary": summary, "folder_counts": dict(counts),
        "image_errors": errors, "exact_duplicates": duplicates,
        "scope": "File SHA-256 equality only. Rotated, cropped, recompressed or other near-duplicate images are not detected. Labels are not assessed.",
    }
    directory = project / "data/audit"
    # parents=True 连同上级目录一起创建；exist_ok=True 允许目录已经存在。
    directory.mkdir(parents=True, exist_ok=True)
    # 时间精确到微秒，让每次运行生成不同名称的报告。
    output = directory / f"apple-audit-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    # x 模式只创建新文件，已有同名报告时不会覆盖。UTF-8 支持中文路径。
    with output.open("x", encoding="utf-8") as handle:
        # indent=2 让 JSON 分行缩进，ensure_ascii=False 让中文正常显示。
        json.dump(report, handle, ensure_ascii=False, indent=2)
    # 在终端显示摘要和报告位置；完整文件明细请打开 JSON 查看。
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Report: {output}")
    print("Finished. No dataset files were changed.")
    return report  # 便于其他 Python 程序调用本函数后取得检查结果。


if __name__ == "__main__":
    # 直接运行本文件才执行检查；被其他程序 import 时不自动执行。
    # __file__ 是本脚本的位置；resolve().parent 得到它所在的项目目录。
    audit(Path(__file__).resolve().parent)

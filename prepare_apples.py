from pathlib import Path, PurePosixPath
from collections import Counter
import zipfile


def extract_apples(project):
    source = project / "data/raw/fruits-original.zip"
    output = project / "data/raw/apples"
    counts = Counter()
    with zipfile.ZipFile(source) as archive:
        selected = []
        destinations = set()
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if (info.is_dir() or len(parts) != 4
                    or parts[0] != "dataset"
                    or parts[1] not in {"train", "test"}
                    or parts[2] not in {"freshapples", "rottenapples"}):
                continue
            if parts[3] in {"", ".", ".."} or "\\" in parts[3]:
                raise RuntimeError(f"Unsafe filename: {info.filename}")
            target = output.joinpath(*parts[1:])
            if target in destinations:
                raise RuntimeError(f"Duplicate destination: {target}")
            destinations.add(target)
            selected.append((info, target))
        if not selected:
            raise RuntimeError("No apple images found; check the archive layout.")
        for number, (info, target) in enumerate(selected, 1):
            data = archive.read(info)  # zipfile checks each member's CRC.
            if target.exists():
                if not target.is_file() or target.read_bytes() != data:
                    raise RuntimeError(f"Existing file differs; preserved: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".extract-part")
                temporary.write_bytes(data)
                temporary.replace(target)
            counts[str(target.parent.relative_to(output))] += 1
            if number % 500 == 0:
                print(f"Verified: {number}/{len(selected)}", flush=True)
    for folder, count in sorted(counts.items()):
        print(f"{folder}: {count}")
    print(f"SUCCESS: {sum(counts.values())} apple images checked and extracted.")
    print(f"Output: {output}")


if __name__ == "__main__":
    extract_apples(Path(__file__).resolve().parent)

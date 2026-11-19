from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


LABELS = ("fresh_banana", "normal_banana", "rotten_banana")
STAGE_NAMES = ("fresh", "normal", "rotten")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def label_to_stage(label: str) -> int:
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")
    return LABELS.index(label)


def deterministic_split(image_id: str, label: str, seed: int = 18) -> str:
    # The label is accepted for API clarity, but the source image alone controls
    # the split. This keeps every annotated object from one photograph together.
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "validation"
    return "test"


def _find_image(images: Path, stem: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = images / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image for annotation {stem}")


def _difference_hash(array: np.ndarray) -> int:
    gray = array.mean(axis=2)
    small = Image.fromarray(gray.astype(np.uint8)).resize((9, 8), Image.Resampling.BILINEAR)
    values = np.asarray(small)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.ravel():
        result = (result << 1) | int(bit)
    return result


def _visual_groups(parsed: list[tuple[str, list]], images: Path,
                   hash_threshold: int = 2, mse_threshold: float = 0.0025) -> dict[str, str]:
    ids = [image_id for image_id, _ in parsed]
    crop_previews: list[tuple[int, np.ndarray, int]] = []
    for source_index, (image_id, objects) in enumerate(parsed):
        with Image.open(_find_image(images, image_id)) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            for _, (xmin, ymin, xmax, ymax) in objects:
                box = (max(0, xmin), max(0, ymin), min(image.width, xmax), min(image.height, ymax))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                preview = np.asarray(image.crop(box).resize((64, 64), Image.Resampling.BILINEAR), dtype=np.uint8)
                crop_previews.append((source_index, preview, _difference_hash(preview)))

    parent = list(range(len(ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for left in range(len(crop_previews)):
        left_source, left_preview, left_hash = crop_previews[left]
        for right in range(left):
            right_source, right_preview, right_hash = crop_previews[right]
            if left_source == right_source or (left_hash ^ right_hash).bit_count() > hash_threshold:
                continue
            mse = np.mean((left_preview.astype(np.float32) - right_preview.astype(np.float32)) ** 2) / (255.0 ** 2)
            if mse <= mse_threshold:
                union(left_source, right_source)
    members = defaultdict(list)
    for index, image_id in enumerate(ids):
        members[find(index)].append(image_id)
    representatives = {
        root: min(group, key=lambda value: (int(value) if value.isdigit() else float("inf"), value))
        for root, group in members.items()
    }
    return {image_id: representatives[find(index)] for index, image_id in enumerate(ids)}


def prepare_voc(source: str | Path, output: str | Path, seed: int = 18) -> dict:
    source, output = Path(source), Path(output)
    annotations, images = source / "Annotations", source / "JPEGImages"
    if not annotations.is_dir() or not images.is_dir():
        raise ValueError("Source must contain Annotations/ and JPEGImages/")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    labels_by_image: dict[str, set[str]] = defaultdict(set)

    parsed = []
    for annotation in sorted(annotations.glob("*.xml"), key=lambda p: p.stem):
        root = ET.parse(annotation).getroot()
        objects = []
        for obj in root.findall("object"):
            label = (obj.findtext("name") or "").strip()
            if label not in LABELS:
                continue
            box = obj.find("bndbox")
            coords = tuple(int(float(box.findtext(name, "0"))) for name in ("xmin", "ymin", "xmax", "ymax"))
            objects.append((label, coords))
            labels_by_image[annotation.stem].add(label)
        if objects:
            parsed.append((annotation.stem, objects))

    visual_group = _visual_groups(parsed, images)

    for image_id, objects in parsed:
        image_path = _find_image(images, image_id)
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            for object_index, (label, (xmin, ymin, xmax, ymax)) in enumerate(objects):
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(image.width, xmax), min(image.height, ymax)
                if xmax <= xmin or ymax <= ymin:
                    continue
                stage = label_to_stage(label)
                group_id = visual_group[image_id]
                split = deterministic_split(group_id, label, seed)
                relative = Path(split) / STAGE_NAMES[stage] / f"{image_id}_{object_index:02d}.jpg"
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.crop((xmin, ymin, xmax, ymax)).save(destination, quality=92)
                rows.append({
                    "path": relative.as_posix(), "label": label, "stage": stage,
                    "split": split, "source_image": image_id, "visual_group": group_id,
                })

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    counts = Counter((row["split"], row["label"]) for row in rows)
    summary = {
        "source": str(source), "seed": seed, "objects": len(rows),
        "source_images": len(parsed),
        "visual_groups": len(set(visual_group.values())),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counts.items())},
    }
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "stage", "split", "source_image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    split_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_groups[row.get("visual_group", row["source_image"])].add(row["split"])
        label_to_stage(row["label"])
    leaked = [group for group, splits in split_groups.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Source-image leakage across splits: {leaked[:5]}")
    return rows


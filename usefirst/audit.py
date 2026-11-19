from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .data import read_manifest


def _decoded_hash(path: Path) -> str:
    with Image.open(path) as image:
        array = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            ImageOps.exif_transpose(image).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0


def _difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        array = np.asarray(gray)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def audit_manifest(manifest: str | Path, output: str | Path | None = None, near_threshold: int = 2,
                   pixel_mse_threshold: float = 0.0025) -> dict:
    manifest = Path(manifest)
    rows = read_manifest(manifest)
    enriched = []
    for row in rows:
        path = manifest.parent / row["path"]
        enriched.append({
            **row,
            "byte_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "decoded_hash": _decoded_hash(path),
            "difference_hash": _difference_hash(path),
        })

    def cross_split_duplicates(field: str) -> list[dict]:
        groups = defaultdict(list)
        for row in enriched:
            groups[row[field]].append(row)
        return [
            {"hash": key, "paths": [row["path"] for row in group], "splits": sorted({row["split"] for row in group})}
            for key, group in groups.items() if len({row["split"] for row in group}) > 1
        ]

    reference = [row for row in enriched if row["split"] != "test"]
    tests = [row for row in enriched if row["split"] == "test"]
    near_pairs = []
    for test in tests:
        best = min(reference, key=lambda row: (test["difference_hash"] ^ row["difference_hash"]).bit_count())
        distance = (test["difference_hash"] ^ best["difference_hash"]).bit_count()
        if distance <= near_threshold:
            mse = float(np.mean((_pixels(manifest.parent / test["path"]) - _pixels(manifest.parent / best["path"])) ** 2))
            if mse <= pixel_mse_threshold:
                near_pairs.append({
                    "test": test["path"], "reference": best["path"],
                    "hamming_distance": distance, "resized_rgb_mse": mse,
                })

    byte_duplicates = cross_split_duplicates("byte_hash")
    decoded_duplicates = cross_split_duplicates("decoded_hash")
    report = {
        "objects": len(rows), "difference_hash_threshold": near_threshold,
        "resized_rgb_mse_threshold": pixel_mse_threshold,
        "cross_split_exact_byte_groups": len(byte_duplicates),
        "cross_split_decoded_pixel_groups": len(decoded_duplicates),
        "test_crops_with_visual_similarity_candidate": len(near_pairs),
        "visual_similarity_candidates": near_pairs[:100],
        "candidate_list_truncated": len(near_pairs) > 100,
    }
    if output:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


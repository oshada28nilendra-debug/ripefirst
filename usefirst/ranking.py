from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from .model import predict


def priority_score(visible_score: float, storage_days: float | None = None, storage_weight: float = 0.15) -> float:
    visible_score = min(100.0, max(0.0, float(visible_score)))
    if storage_days is None:
        return visible_score
    if storage_days < 0:
        raise ValueError("storage_days cannot be negative")
    storage_pressure = min(storage_days / 7.0, 1.0) * 100.0
    return (1.0 - storage_weight) * visible_score + storage_weight * storage_pressure


def rank_manifest(model: str | Path, manifest: str | Path, storage_weight: float = 0.15) -> list[dict]:
    manifest = Path(manifest)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"item_id", "image"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Ranking CSV must contain item_id and image")
    results = []
    for row in rows:
        image = Path(row["image"])
        if not image.is_absolute():
            image = manifest.parent / image
        result = predict(model, image)
        storage_days = float(row["storage_days"]) if row.get("storage_days", "").strip() else None
        result.update({
            "item_id": row["item_id"], "storage_days": storage_days,
            "priority_score": priority_score(result["visible_deterioration_score"], storage_days, storage_weight),
        })
        results.append(result)
    results.sort(key=lambda item: (item["uncertain"], -item["priority_score"]))
    for position, item in enumerate(results, 1):
        item["rank"] = position
        item["action"] = "review_image" if item["uncertain"] else "use_first"
    return results


def append_observation(store: str | Path, model: str | Path, image: str | Path, item_id: str, observed_at: str) -> dict:
    timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    store = Path(store)
    observations = json.loads(store.read_text()) if store.exists() else []
    if any(row["item_id"] == item_id and row["observed_at"] == observed_at for row in observations):
        raise ValueError("Duplicate item_id and observed_at")
    result = predict(model, image)
    previous = [row for row in observations if row["item_id"] == item_id]
    previous.sort(key=lambda row: row["observed_at"])
    trend = None
    if previous:
        before = datetime.fromisoformat(previous[-1]["observed_at"].replace("Z", "+00:00"))
        elapsed = (timestamp - before).total_seconds() / 86400
        if elapsed <= 0:
            raise ValueError("observed_at must be later than the previous observation")
        trend = (result["visible_deterioration_score"] - previous[-1]["visible_deterioration_score"]) / elapsed
    record = {
        "item_id": item_id, "observed_at": observed_at, "image": str(image),
        "stage": result["stage"], "visible_deterioration_score": result["visible_deterioration_score"],
        "confidence": result["confidence"], "uncertain": result["uncertain"],
        "score_change_per_day": trend,
    }
    observations.append(record)
    store.parent.mkdir(parents=True, exist_ok=True)
    temporary = store.with_suffix(store.suffix + ".tmp")
    temporary.write_text(json.dumps(observations, indent=2) + "\n")
    temporary.replace(store)
    return record


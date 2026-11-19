from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw


def create_demo(output: str | Path) -> Path:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    colors = [(101, 168, 42), (224, 190, 36), (132, 87, 45)]
    names = ["fresh", "normal", "rotten"]
    rows = []
    for index, (name, color) in enumerate(zip(names, colors)):
        image = Image.new("RGB", (360, 260), (238, 235, 220))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((55, 75, 305, 175), radius=42, fill=color, outline=(70, 55, 35), width=4)
        if index:
            for spot in range(index * 6):
                x = 90 + (spot * 37) % 190; y = 95 + (spot * 23) % 60
                draw.ellipse((x, y, x + 12 + index * 3, y + 8 + index * 2), fill=(78, 48, 30))
        path = output / f"synthetic_{name}.png"; image.save(path)
        rows.append({"item_id": f"demo-{index + 1}", "image": path.name, "storage_days": str(index * 2)})
    manifest = output / "items.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return manifest


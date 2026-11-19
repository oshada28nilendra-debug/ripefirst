from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


FEATURE_VERSION = "color-texture-v1"


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    valid = delta > 1e-7
    red = (maximum == rgb[:, :, 0]) & valid
    green = (maximum == rgb[:, :, 1]) & valid
    blue = (maximum == rgb[:, :, 2]) & valid
    hue[red] = ((rgb[:, :, 1] - rgb[:, :, 2])[red] / delta[red]) % 6
    hue[green] = (rgb[:, :, 2] - rgb[:, :, 0])[green] / delta[green] + 2
    hue[blue] = (rgb[:, :, 0] - rgb[:, :, 1])[blue] / delta[blue] + 4
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-7)
    return np.stack([hue, saturation, maximum], axis=2)


def _hist(values: np.ndarray, bins: int = 16) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return counts.astype(np.float32) / max(1, values.size)


def extract_features(image: str | Path | np.ndarray) -> np.ndarray:
    rgb = load_rgb(image) if isinstance(image, (str, Path)) else np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, 1.0))
    hue, saturation, value = hsv.transpose(2, 0, 1)

    histograms = np.concatenate([*(_hist(rgb[:, :, c]) for c in range(3)),
                                 *(_hist(hsv[:, :, c]) for c in range(3))])
    channel_stats = np.concatenate([
        rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1)),
        hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1)),
        np.quantile(hsv.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).reshape(-1),
    ])
    colorful = saturation > 0.18
    color_fractions = np.array([
        np.mean(colorful & (hue >= 0.18) & (hue <= 0.48)),
        np.mean(colorful & (hue >= 0.10) & (hue < 0.18) & (value > 0.35)),
        np.mean(colorful & (hue >= 0.02) & (hue < 0.12) & (value < 0.72)),
        np.mean(value < 0.25),
        np.mean(value > 0.82),
        np.mean(saturation < 0.12),
    ], dtype=np.float32)

    gray = rgb.mean(axis=2)
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    gradients = np.concatenate([np.abs(dx).ravel(), np.abs(dy).ravel()])
    texture = np.array([
        gradients.mean(), gradients.std(),
        *np.quantile(gradients, [0.5, 0.9, 0.99]),
        np.var(dx[:, 1:] - dx[:, :-1]) if dx.shape[1] > 1 else 0.0,
        np.var(dy[1:, :] - dy[:-1, :]) if dy.shape[0] > 1 else 0.0,
    ], dtype=np.float32)
    return np.concatenate([histograms, channel_stats, color_fractions, texture]).astype(np.float32)


def quality_flags(image: str | Path | np.ndarray) -> list[str]:
    rgb = load_rgb(image) if isinstance(image, (str, Path)) else np.asarray(image, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    value = rgb.max(axis=2)
    gray = rgb.mean(axis=2)
    gradient = np.concatenate([np.abs(np.diff(gray, axis=0)).ravel(), np.abs(np.diff(gray, axis=1)).ravel()])
    flags: list[str] = []
    if value.mean() < 0.18:
        flags.append("too_dark")
    if value.mean() > 0.92:
        flags.append("too_bright")
    if np.quantile(gradient, 0.9) < 0.018:
        flags.append("possibly_blurry")
    if min(rgb.shape[:2]) < 80:
        flags.append("low_resolution")
    return flags


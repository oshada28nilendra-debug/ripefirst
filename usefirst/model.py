from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import STAGE_NAMES, read_manifest
from .features import FEATURE_VERSION, extract_features, load_rgb, quality_flags


def _load_split(manifest_path: Path, rows: list[dict], split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    selected = [row for row in rows if row["split"] == split]
    root = manifest_path.parent
    x = np.stack([extract_features(root / row["path"]) for row in selected])
    y = np.array([int(row["stage"]) for row in selected], dtype=np.int64)
    return x, y, selected


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    expected = probability @ np.arange(len(STAGE_NAMES))
    confidence = probability.max(axis=1)
    entropy = -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)), axis=1) / np.log(len(STAGE_NAMES))
    accepted = (confidence >= 0.60) & (entropy <= 0.72)
    errors = prediction != y_true
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "ordinal_mae_stages": float(mean_absolute_error(y_true, expected)),
        "log_loss": float(log_loss(y_true, probability, labels=np.arange(len(STAGE_NAMES)))),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=np.arange(len(STAGE_NAMES))).tolist(),
        "uncertainty_review_rate": float(1.0 - accepted.mean()),
        "accepted_accuracy": float(accuracy_score(y_true[accepted], prediction[accepted])) if accepted.any() else None,
        "error_review_recall": float((~accepted & errors).sum() / errors.sum()) if errors.any() else None,
    }


def _perturbed_features(manifest_path: Path, rows: list[dict], mode: str) -> np.ndarray:
    features = []
    for row in rows:
        rgb = load_rgb(manifest_path.parent / row["path"])
        if mode == "dark_0.7":
            changed = rgb * 0.7
        elif mode == "bright_1.3":
            changed = rgb * 1.3
        elif mode == "warm_cast":
            changed = rgb * np.array([1.12, 1.0, 0.88], dtype=np.float32)
        else:
            raise ValueError(f"Unknown perturbation: {mode}")
        features.append(extract_features(np.clip(changed, 0.0, 1.0)))
    return np.stack(features)


def _training_features(manifest_path: Path, rows: list[dict], x_original: np.ndarray, y_original: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    modes = ("dark_0.7", "bright_1.3", "warm_cast")
    augmented = [x_original]
    for mode in modes:
        augmented.append(_perturbed_features(manifest_path, rows, mode))
    return np.concatenate(augmented), np.tile(y_original, len(augmented))


def fit(manifest: str | Path, output: str | Path, report: str | Path | None = None, seed: int = 18) -> dict:
    manifest, output = Path(manifest), Path(output)
    rows = read_manifest(manifest)
    x_train, y_train, train_rows = _load_split(manifest, rows, "train")
    x_validation, y_validation, validation_rows = _load_split(manifest, rows, "validation")
    x_test, y_test, test_rows = _load_split(manifest, rows, "test")
    x_train_augmented, y_train_augmented = _training_features(manifest, train_rows, x_train, y_train)
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
    }
    validation_results = {}
    for name, candidate in candidates.items():
        candidate.fit(x_train_augmented, y_train_augmented)
        validation_results[name] = _metrics(y_validation, candidate.predict_proba(x_validation))
    winner = max(candidates, key=lambda name: (validation_results[name]["macro_f1"], -validation_results[name]["log_loss"]))
    model = candidates[winner]
    x_validation_augmented, y_validation_augmented = _training_features(
        manifest, validation_rows, x_validation, y_validation)
    x_fit = np.concatenate([x_train_augmented, x_validation_augmented])
    y_fit = np.concatenate([y_train_augmented, y_validation_augmented])
    model.fit(x_fit, y_fit)
    test_probability = model.predict_proba(x_test)
    robustness = {}
    for mode in ("dark_0.7", "bright_1.3", "warm_cast"):
        robustness[mode] = _metrics(y_test, model.predict_proba(_perturbed_features(manifest, test_rows, mode)))
    artifact = {
        "model": model, "classes": list(STAGE_NAMES), "feature_version": FEATURE_VERSION,
        "seed": seed, "selected_model": winner,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    results = {
        "dataset": {
            "objects": len(rows), "source_images": len({r["source_image"] for r in rows}),
            "train_objects": len(train_rows), "validation_objects": len(validation_rows), "test_objects": len(test_rows),
            "test_source_images": len({r["source_image"] for r in test_rows}),
            "training_augmentation_modes": ["dark_0.7", "bright_1.3", "warm_cast"],
        },
        "candidate_validation": validation_results,
        "selected_model": winner,
        "test": _metrics(y_test, test_probability),
        "digital_perturbation_stress_tests": robustness,
    }
    if report:
        report = Path(report); report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(results, indent=2) + "\n")
    return results


def load_model(path: str | Path) -> dict:
    artifact = joblib.load(path)
    required = {"model", "classes", "feature_version"}
    if not required.issubset(artifact) or artifact["feature_version"] != FEATURE_VERSION:
        raise ValueError("Unsupported or incompatible model artifact")
    return artifact


def predict(model_path: str | Path, image_path: str | Path) -> dict:
    artifact = load_model(model_path)
    probability = artifact["model"].predict_proba(extract_features(image_path)[None])[0]
    classes = artifact["classes"]
    expected = float(probability @ np.arange(len(classes)))
    confidence = float(probability.max())
    entropy = float(-np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))) / np.log(len(classes)))
    flags = quality_flags(image_path)
    uncertain = confidence < 0.60 or entropy > 0.72 or bool(flags)
    return {
        "image": str(image_path), "stage": classes[int(probability.argmax())],
        "probabilities": {name: float(value) for name, value in zip(classes, probability)},
        "expected_stage": expected, "visible_deterioration_score": 50.0 * expected,
        "confidence": confidence, "normalized_entropy": entropy,
        "uncertain": uncertain, "quality_flags": flags,
        "interpretation": "visible appearance only; not a safety or shelf-life estimate",
    }


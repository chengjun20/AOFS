"""Strict loading and aggregation for multi-seed AOFS OBB results."""

import json
import statistics
from pathlib import Path


def summarize_runs(runs):
    """Return mean and sample standard deviation for at least three unique seeds."""
    if len(runs) < 3:
        raise ValueError("at least three seed results are required")
    normalized = [dict(run) for run in runs]
    normalized.sort(key=lambda run: int(run["seed"]))
    seeds = [int(run["seed"]) for run in normalized]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed values must be unique")
    values = [float(run["novel_map50"]) for run in normalized]
    return {
        "seeds": seeds,
        "novel_map50_mean": statistics.mean(values),
        "novel_map50_std": statistics.stdev(values),
        "runs": normalized,
    }


def _find_run_manifest(metrics_path):
    metrics_path = Path(metrics_path).resolve()
    for parent in metrics_path.parents:
        candidate = parent / "run_manifest.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No parent run_manifest.json found for {metrics_path}")


def load_metric_run(metrics_path):
    """Combine one OBB report with identity from its parent training run."""
    metrics_path = Path(metrics_path).resolve()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest_path = _find_run_manifest(metrics_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity") or {}
    required_identity = ("profile", "dataset", "shot", "seed", "stage")
    missing = [key for key in required_identity if key not in identity]
    if missing:
        raise ValueError(
            f"Missing identity field(s) in {manifest_path}: {', '.join(missing)}"
        )
    if "novel_map50" not in metrics:
        raise ValueError(f"Missing novel_map50 in {metrics_path}")
    return {
        **{key: identity[key] for key in required_identity},
        "shot": int(identity["shot"]),
        "seed": int(identity["seed"]),
        "novel_map50": float(metrics["novel_map50"]),
        "metrics_file": str(metrics_path),
        "manifest_file": str(manifest_path),
    }


def summarize_metric_files(metric_paths):
    """Load OBB metric files and reject accidental cross-experiment aggregation."""
    runs = [load_metric_run(path) for path in metric_paths]
    for field in ("profile", "dataset", "shot", "stage"):
        values = {run[field] for run in runs}
        if len(values) != 1:
            raise ValueError(f"Cannot mix {field} values: {sorted(values, key=str)}")
    summary = summarize_runs(runs)
    summary.update(
        {
            "profile": runs[0]["profile"],
            "dataset": runs[0]["dataset"],
            "shot": runs[0]["shot"],
            "stage": runs[0]["stage"],
        }
    )
    return summary

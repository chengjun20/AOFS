"""Portable experiment configuration and run identity helpers."""

import json
import math
import os
import re
import subprocess
from pathlib import Path


PATH_KEYS = {"train", "valid", "meta", "support_root"}
VARIABLE_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_data_options(options, data_root=None, overrides=None):
    """Apply explicit overrides, expand environment variables, and normalize paths."""
    resolved = dict(options)
    if data_root:
        root = Path(os.path.expandvars(str(data_root))).expanduser()
        resolved["train"] = str(root / "training.txt")
        resolved["valid"] = str(root / "evaluation.txt")
    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value not in (None, "")})

    for key, original_value in tuple(resolved.items()):
        if not isinstance(original_value, str):
            continue
        value = os.path.expanduser(os.path.expandvars(original_value))
        missing = VARIABLE_PATTERN.findall(value)
        if missing:
            raise ValueError(f"Unresolved environment variable(s) in {key}: {', '.join(missing)}")
        resolved[key] = os.path.normpath(value) if key in PATH_KEYS else value
    return resolved


def support_label_path(support_root, class_name, image_path):
    """Return the class-specific support label for an image in a generated split."""
    return os.path.normpath(
        str(Path(support_root) / str(class_name) / (Path(image_path).stem + ".txt"))
    )


def override_dataset_root(dataset_definition, data_root):
    """Return a dataset YAML mapping with an optional portable root override."""
    resolved = dict(dataset_definition)
    if data_root:
        value = os.path.expanduser(os.path.expandvars(str(data_root)))
        missing = VARIABLE_PATTERN.findall(value)
        if missing:
            raise ValueError(
                "Unresolved environment variable(s) in dataset root: " + ", ".join(missing)
            )
        resolved["path"] = os.path.normpath(value)
    return resolved


def compute_tuning_epochs(max_epoch, repeat):
    """Convert the author's max-iteration style setting to whole training epochs."""
    repeat = int(repeat)
    if repeat <= 0:
        raise ValueError("repeat must be greater than zero")
    return int(math.ceil(int(max_epoch) / repeat))


def should_validate(epoch, epochs, noval, period):
    """Validate periodically, while always preserving final-epoch validation."""
    final_epoch = epoch + 1 == epochs
    if final_epoch:
        return True
    if noval:
        return False
    return period > 0 and (epoch + 1) % period == 0


def build_experiment_identity(profile, dataset, shot, seed, stage):
    """Build the fields that must match when resuming an interrupted run."""
    return {
        "profile": str(profile),
        "dataset": str(dataset),
        "shot": int(shot),
        "seed": int(seed),
        "stage": str(stage),
    }


def validate_resume_identity(expected, actual):
    """Reject checkpoints that belong to a different experiment target."""
    if not actual:
        raise ValueError("Checkpoint has no experiment identity; use it as initial weights, not --resume")
    differences = [key for key, value in expected.items() if actual.get(key) != value]
    if differences:
        raise ValueError("Resume identity mismatch: " + ", ".join(differences))


def write_run_manifest(path, identity, options, data_options):
    """Persist resolved parameters and Git provenance next to a training run."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None

    payload = {
        "identity": identity,
        "options": vars(options),
        "data_options": data_options,
        "git_revision": revision,
        "git_dirty": dirty,
    }
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

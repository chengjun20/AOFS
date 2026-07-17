"""Deterministic support-instance selection for AOFS few-shot experiments."""

import collections
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from aofs.datasets import get_dataset_spec


@dataclass(frozen=True)
class Candidate:
    image_path: str
    class_name: str
    polygon: tuple
    difficult: str

    def to_dota_line(self):
        return " ".join((*self.polygon, self.class_name, self.difficult))


def select_candidates(candidates, shot, seed, max_per_image):
    """Select exactly ``shot`` instances under an optional per-image cap."""
    shot = int(shot)
    if shot <= 0:
        raise ValueError("shot must be greater than zero")
    if max_per_image is not None and int(max_per_image) <= 0:
        raise ValueError("max_per_image must be greater than zero")

    rng = random.Random(int(seed))
    ordered = list(candidates)
    rng.shuffle(ordered)
    if max_per_image is None:
        if len(ordered) < shot:
            raise ValueError(f"requested {shot} instances but only {len(ordered)} are available")
        return ordered[:shot]

    max_per_image = int(max_per_image)
    eligible_images = {item.image_path for item in ordered}
    if len(eligible_images) < math.ceil(shot / max_per_image):
        raise ValueError(
            f"requested {shot} instances with max_per_image={max_per_image}, "
            f"but only {len(eligible_images)} eligible images are available"
        )

    counts = collections.Counter()
    selected = []
    for item in ordered:
        if counts[item.image_path] >= max_per_image:
            continue
        selected.append(item)
        counts[item.image_path] += 1
        if len(selected) == shot:
            return selected
    raise ValueError(f"unable to select {shot} instances")


def _resolve_image_path(data_root, value):
    image_path = Path(value.strip()).expanduser()
    if not image_path.is_absolute():
        image_path = Path(data_root) / image_path
    return image_path.resolve()


def _find_label_path(data_root, image_path):
    data_root = Path(data_root).resolve()
    candidates = [
        data_root / "labels" / f"{image_path.stem}.txt",
        data_root / "training" / "labels" / f"{image_path.stem}.txt",
        data_root / "training" / "labelTxt" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labels" / f"{image_path.stem}.txt",
        image_path.parent.parent / "labelTxt" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No label file found for {image_path}; checked: {candidates}")


def load_candidates(data_root, dataset_name="nwpu"):
    """Read DOTA-style training annotations into per-class candidates."""
    data_root = Path(data_root).resolve()
    training_list = data_root / "training.txt"
    if not training_list.is_file():
        raise FileNotFoundError(f"Training list not found: {training_list}")
    spec = get_dataset_spec(dataset_name)
    candidates = {class_name: [] for class_name in spec.classes}

    for value in training_list.read_text(encoding="utf-8").splitlines():
        if not value.strip():
            continue
        image_path = _resolve_image_path(data_root, value)
        label_path = _find_label_path(data_root, image_path)
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 9 or fields[8] not in candidates:
                continue
            if len(fields[:8]) != 8:
                raise ValueError(f"Invalid polygon at {label_path}:{line_number}")
            difficult = fields[9] if len(fields) > 9 else "0"
            candidate = Candidate(str(image_path), fields[8], tuple(fields[:8]), difficult)
            candidates[candidate.class_name].append(candidate)
    return candidates


def generate_split(data_root, output_root, profile, shot, seed, max_per_image=None):
    """Generate seed-specific support labels, meta lists, and an audit report."""
    if profile not in {"paper", "robust"}:
        raise ValueError(f"Unsupported profile: {profile}")
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    support_root = output_root / "support_labels"
    lists_root = output_root / "lists"
    support_root.mkdir(parents=True, exist_ok=True)
    lists_root.mkdir(parents=True, exist_ok=True)

    spec = get_dataset_spec("nwpu")
    available = load_candidates(data_root, "nwpu")
    audit_classes = {}
    meta_lines = []

    for class_index, class_name in enumerate(spec.classes):
        selected = select_candidates(
            available[class_name],
            shot=shot,
            seed=int(seed) + class_index * 1000003,
            max_per_image=max_per_image,
        )
        class_root = support_root / class_name
        class_root.mkdir(parents=True, exist_ok=True)
        by_image = collections.defaultdict(list)
        for candidate in selected:
            by_image[candidate.image_path].append(candidate)
        for image_path, image_candidates in by_image.items():
            label_path = class_root / f"{Path(image_path).stem}.txt"
            label_path.write_text(
                "\n".join(item.to_dota_line() for item in image_candidates) + "\n",
                encoding="utf-8",
            )

        list_path = lists_root / f"{class_name}.txt"
        image_paths = list(by_image.keys())
        list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        meta_lines.append(f"{class_name} {list_path}")
        image_counts = collections.Counter(item.image_path for item in selected)
        audit_classes[class_name] = {
            "instances": len(selected),
            "unique_images": len(image_counts),
            "max_instances_per_image": max(image_counts.values()),
            "selected": [
                {
                    "image_path": item.image_path,
                    "polygon": list(item.polygon),
                    "difficult": item.difficult,
                }
                for item in selected
            ],
        }

    meta_path = output_root / "meta.txt"
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    audit = {
        "dataset": "nwpu",
        "profile": profile,
        "shot": int(shot),
        "seed": int(seed),
        "max_per_image": None if max_per_image is None else int(max_per_image),
        "classes": audit_classes,
    }
    canonical = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    audit["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    audit_path = output_root / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return {"meta": meta_path, "support_root": support_root, "audit": audit_path}

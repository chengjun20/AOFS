"""Dataset-aware oriented bounding-box evaluation adapters."""

import json
from pathlib import Path

from aofs.datasets import get_dataset_spec


def aggregate_group_aps(per_class, class_names):
    """Average AP over the declared group, counting absent predictions as zero."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    return sum(float(per_class.get(name, 0.0)) for name in class_names) / len(class_names)


def count_ground_truths(annopath, imagesetfile, dataset_name):
    """Count non-difficult polygon ground truths for every registered class."""
    spec = get_dataset_spec(dataset_name)
    counts = {class_name: 0 for class_name in spec.classes}
    image_ids = Path(imagesetfile).read_text(encoding="utf-8").splitlines()
    for image_id in image_ids:
        if not image_id.strip():
            continue
        annotation_file = Path(str(annopath).format(image_id.strip()))
        for line in annotation_file.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split()
            if len(fields) < 9 or fields[8] not in counts:
                continue
            difficult = fields[9] if len(fields) > 9 else "0"
            if difficult in {"0", "0.0"}:
                counts[fields[8]] += 1
    return counts


def convert_json_predictions(json_file, detection_dir, dataset_name):
    """Convert AOFS polygon JSON predictions to DOTA Task1 per-class files."""
    spec = get_dataset_spec(dataset_name)
    json_file = Path(json_file)
    detection_dir = Path(detection_dir)
    detection_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}
    for class_name in spec.classes:
        output_file = detection_dir / f"Task1_{class_name}.txt"
        output_file.write_text("", encoding="utf-8")
        output_files[class_name] = output_file

    predictions = json.loads(json_file.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Prediction JSON must contain a list: {json_file}")

    handles = {}
    try:
        for prediction in predictions:
            category_id = int(prediction["category_id"])
            if not 1 <= category_id <= len(spec.classes):
                raise ValueError(
                    f"category_id {category_id} is outside 1..{len(spec.classes)} for {dataset_name}"
                )
            polygon = prediction["poly"]
            if len(polygon) != 8:
                raise ValueError(f"Expected 8 polygon coordinates, got {len(polygon)}")
            class_name = spec.classes[category_id - 1]
            handle = handles.get(class_name)
            if handle is None:
                handle = output_files[class_name].open("a", encoding="utf-8")
                handles[class_name] = handle
            values = [prediction["file_name"], prediction["score"], *polygon]
            handle.write(" ".join(str(value) for value in values) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return detection_dir


def evaluate_detection_files(detection_dir, annopath, imagesetfile, dataset_name, voc_eval_fn):
    """Evaluate every registered class without shrinking group denominators."""
    spec = get_dataset_spec(dataset_name)
    detection_dir = Path(detection_dir)
    detpath = str(detection_dir / "Task1_{:s}.txt")
    per_class = {}
    prediction_counts = {}

    for class_name in spec.classes:
        detection_file = detection_dir / f"Task1_{class_name}.txt"
        prediction_counts[class_name] = (
            len(detection_file.read_text(encoding="utf-8").splitlines())
            if detection_file.exists()
            else 0
        )
        if not detection_file.exists() or detection_file.stat().st_size == 0:
            per_class[class_name] = 0.0
            continue
        _, _, ap = voc_eval_fn(
            detpath,
            annopath,
            imagesetfile,
            class_name,
            ovthresh=0.5,
            use_07_metric=True,
        )
        per_class[class_name] = float(ap)

    return {
        "dataset": str(dataset_name).lower(),
        "per_class": per_class,
        "prediction_counts": prediction_counts,
        "novel_map50": aggregate_group_aps(per_class, spec.novel),
        "base_map50": aggregate_group_aps(per_class, spec.base),
        "all_map50": aggregate_group_aps(per_class, spec.classes),
    }


def evaluate_obb_predictions(
    json_file,
    annopath,
    imagesetfile,
    dataset_name,
    output_dir=None,
    voc_eval_fn=None,
):
    """Convert JSON, run VOC 2007 polygon AP, and persist a machine-readable report."""
    json_file = Path(json_file)
    output_dir = Path(output_dir) if output_dir else json_file.parent / f"{json_file.stem}_Txt"
    convert_json_predictions(json_file, output_dir, dataset_name)

    if voc_eval_fn is None:
        from DOTA_devkit.dota_evaluation_task1 import voc_eval as voc_eval_fn

    metrics = evaluate_detection_files(
        output_dir,
        annopath,
        imagesetfile,
        dataset_name,
        voc_eval_fn,
    )
    metrics["positive_counts"] = count_ground_truths(
        annopath, imagesetfile, dataset_name
    )
    metrics["prediction_json"] = str(json_file.resolve())
    metrics_file = output_dir / "obb_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metrics["metrics_file"] = str(metrics_file.resolve())
    return metrics

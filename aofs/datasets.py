"""Canonical dataset class orders and few-shot splits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    classes: tuple
    novel: tuple
    base: tuple


NWPU_CLASSES = (
    "airplane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "vehicle",
)
NWPU_NOVEL = ("airplane", "baseball-diamond", "tennis-court")
NWPU = DatasetSpec(
    NWPU_CLASSES,
    NWPU_NOVEL,
    tuple(name for name in NWPU_CLASSES if name not in NWPU_NOVEL),
)

DIOR_CLASSES = (
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
)
DIOR_NOVEL = ("airplane", "baseballfield", "tenniscourt", "trainstation", "windmill")
DIOR = DatasetSpec(
    DIOR_CLASSES,
    DIOR_NOVEL,
    tuple(name for name in DIOR_CLASSES if name not in DIOR_NOVEL),
)

DATASETS = {"nwpu": NWPU, "dior": DIOR}


def get_dataset_spec(name):
    """Return the canonical class order and split for a supported dataset."""
    try:
        return DATASETS[str(name).lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {name}") from error

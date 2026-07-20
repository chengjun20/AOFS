#!/usr/bin/env bash
set -euo pipefail

: "${AOFS_DATA_ROOT:?Set AOFS_DATA_ROOT to the NWPU-R dataset root}"

SHOT="${1:?Usage: $0 <3|5|10> <seed> <device> <base-weight.pt>}"
SEED="${2:?Usage: $0 <3|5|10> <seed> <device> <base-weight.pt>}"
DEVICE="${3:?Usage: $0 <3|5|10> <seed> <device> <base-weight.pt>}"
BASE_WEIGHT="${4:?Usage: $0 <3|5|10> <seed> <device> <base-weight.pt>}"

case "$SHOT" in
  3|5|10) ;;
  *) echo "SHOT must be 3, 5, or 10" >&2; exit 2 ;;
esac
[[ -f "$BASE_WEIGHT" ]] || { echo "Base weight not found: $BASE_WEIGHT" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLITS_ROOT="${AOFS_SPLITS_ROOT:-runs/splits}"
export AOFS_SPLIT_ROOT="${SPLITS_ROOT}/robust/${SHOT}shot/seed_${SEED}"
export AOFS_SEED="$SEED"
if [[ -n "${AOFS_OBB_ANNOPATH:-}" ]]; then
  OBB_ANNOPATH="$AOFS_OBB_ANNOPATH"
else
  OBB_ANNOPATH="${AOFS_DATA_ROOT}/evaluation/labelTxt/{:s}.txt"
fi
OBB_IMAGESET="${AOFS_OBB_IMAGESET:-${AOFS_DATA_ROOT}/imgnamefile.txt}"

"$PYTHON_BIN" tools/gen_fewlist_nwpu.py \
  --data-root "$AOFS_DATA_ROOT" \
  --output-root "$AOFS_SPLIT_ROOT" \
  --profile robust \
  --shot "$SHOT" \
  --seed "$SEED"

"$PYTHON_BIN" train.py \
  --weights "$BASE_WEIGHT" \
  --data data/nwpu_poly.yaml \
  --data-root "$AOFS_DATA_ROOT" \
  --cfgdata "cfg/robust/nwpu_${SHOT}shot.data" \
  --hyp cfg/robust/hyp.finetune_nwpu.yaml \
  --profile robust \
  --stage fewtune \
  --dataset-name nwpu \
  --shot "$SHOT" \
  --seed "$SEED" \
  --batch-size "${AOFS_BATCH_SIZE:-1}" \
  --img "${AOFS_IMAGE_SIZE:-1024}" \
  --workers "${AOFS_WORKERS:-8}" \
  --device "$DEVICE" \
  --project runs/robust \
  --name "nwpu_${SHOT}shot_seed_${SEED}" \
  --patience 500 \
  --val-period "${AOFS_VAL_PERIOD:-10}" \
  --checkpoint-metric obb_novel_map50 \
  --obb-annopath "$OBB_ANNOPATH" \
  --obb-imagesetfile "$OBB_IMAGESET"

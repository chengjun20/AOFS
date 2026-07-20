#!/usr/bin/env bash
set -euo pipefail

: "${AOFS_DATA_ROOT:?Set AOFS_DATA_ROOT to the NWPU-R dataset root}"

PROFILE="${1:?Usage: $0 <paper|robust> <3|5|10> <seed> <device> <weights.pt>}"
SHOT="${2:?Usage: $0 <paper|robust> <3|5|10> <seed> <device> <weights.pt>}"
SEED="${3:?Usage: $0 <paper|robust> <3|5|10> <seed> <device> <weights.pt>}"
DEVICE="${4:?Usage: $0 <paper|robust> <3|5|10> <seed> <device> <weights.pt>}"
WEIGHTS="${5:?Usage: $0 <paper|robust> <3|5|10> <seed> <device> <weights.pt>}"

case "$PROFILE" in paper|robust) ;; *) echo "PROFILE must be paper or robust" >&2; exit 2 ;; esac
case "$SHOT" in 3|5|10) ;; *) echo "SHOT must be 3, 5, or 10" >&2; exit 2 ;; esac
[[ -f "$WEIGHTS" ]] || { echo "Weights not found: $WEIGHTS" >&2; exit 2; }

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLITS_ROOT="${AOFS_SPLITS_ROOT:-runs/splits}"
export AOFS_SPLIT_ROOT="${SPLITS_ROOT}/${PROFILE}/${SHOT}shot/seed_${SEED}"
export AOFS_SEED="$SEED"
if [[ -n "${AOFS_OBB_ANNOPATH:-}" ]]; then
  OBB_ANNOPATH="$AOFS_OBB_ANNOPATH"
else
  OBB_ANNOPATH="${AOFS_DATA_ROOT}/evaluation/labelTxt/{:s}.txt"
fi
OBB_IMAGESET="${AOFS_OBB_IMAGESET:-${AOFS_DATA_ROOT}/imgnamefile.txt}"
PREDICTION_STEM="nwpu_${PROFILE}_${SHOT}shot_seed_${SEED}"
VAL_PROJECT="${AOFS_EVAL_ROOT:-runs/eval}"
VAL_NAME="$PREDICTION_STEM"

"$PYTHON_BIN" tools/gen_fewlist_nwpu.py \
  --data-root "$AOFS_DATA_ROOT" \
  --output-root "$AOFS_SPLIT_ROOT" \
  --profile "$PROFILE" \
  --shot "$SHOT" \
  --seed "$SEED"

"$PYTHON_BIN" val.py \
  --weights "$WEIGHTS" \
  --data data/nwpu_poly.yaml \
  --data-root "$AOFS_DATA_ROOT" \
  --cfgdata "cfg/${PROFILE}/nwpu_${SHOT}shot.data" \
  --batch-size "${AOFS_BATCH_SIZE:-1}" \
  --img "${AOFS_IMAGE_SIZE:-1024}" \
  --device "$DEVICE" \
  --save-json \
  --prediction-stem "$PREDICTION_STEM" \
  --project "$VAL_PROJECT" \
  --name "$VAL_NAME" \
  --exist-ok

"$PYTHON_BIN" DOTA_devkit/dota_evaluation_task1.py \
  --dataset nwpu \
  --base_path "${VAL_PROJECT}/${VAL_NAME}/${PREDICTION_STEM}" \
  --annopath "$OBB_ANNOPATH" \
  --imagesetfile "$OBB_IMAGESET"

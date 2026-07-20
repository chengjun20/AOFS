# AOFS NWPU Base Hyperparameter Separation Design

## Goal

Make the validated NWPU Base-stage optimizer setting explicit and reproducible without changing the AOFS model, the accepted Base checkpoint, or the paper/robust few-shot training settings.

## Evidence and decision

The paper reports SGD with learning rate `0.001`, momentum `0.999`, and weight decay `0.0005`. The author's public NWPU configuration instead uses momentum `0.937` with the same learning rate, weight decay, augmentation, and loss settings.

Server experiments on the same NWPU data and AOFS-S model isolated the practical effect:

- The first 100-epoch Base run used momentum `0.999`; its independently evaluated HBB mAP@0.5 was `0.0886` and HBB mAP@0.5:0.95 was `0.0322`.
- The historical user-produced `exp6` checkpoint used momentum `0.937`; its independently evaluated HBB mAP@0.5 was `0.205` and HBB mAP@0.5:0.95 was `0.0672`.
- A controlled 10-epoch comparison showed that `0.999` learned faster initially (`0.054351` HBB mAP@0.5 versus `0` for `0.937`), so early results alone did not justify changing the setting.
- A controlled 100-epoch run under the current code changed only the Base hyperparameter selection from the paper-profile file to the author's existing `0.937` file. No server source file was edited. The selected checkpoint was epoch 69 with training-time HBB mAP@0.5 `0.276260` and HBB mAP@0.5:0.95 `0.097726`. Independent validation reproduced `0.277` and `0.0979` respectively.

The accepted server Base checkpoint is therefore:

```text
/workspace/AOFS_new/runs/base/nwpu_aofs_s_m0937/weights/best.pt
```

The checkpoint is a generated binary and will not be committed to Git.

## Design

Add a dedicated `cfg/paper/hyp.base_nwpu.yaml` derived from the author's public NWPU hyperparameter file. It will use momentum `0.937` and retain the existing learning rate `0.001`, weight decay `0.0005`, augmentation, and loss settings.

Keep both few-shot files unchanged:

- `cfg/paper/hyp.finetune_nwpu.yaml` remains at momentum `0.999` to preserve the paper-text few-shot profile.
- `cfg/robust/hyp.finetune_nwpu.yaml` remains identical to the paper few-shot optimizer settings so the robust comparison continues to change only support sampling.

Update the Base command in `docs/AOFS_REPRODUCTION.md` to use the dedicated Base file. Document the paper/public-code mismatch, the controlled server evidence, and the distinction between the Base diagnostic HBB metric and the final novel OBB paper metric.

Update `tests/test_profiles.py` so it independently asserts:

- Base momentum is `0.937`.
- Paper and robust few-shot momentum remains `0.999`.
- Base and few-shot files retain learning rate `0.001` and weight decay `0.0005`.

Update `MODIFICATIONS_FROM_UPSTREAM.md` in the same implementation patch with every affected path and the server validation evidence.

## Out of scope

- No changes to `train.py`, `val.py`, model architecture, losses, data loading, support sampling, or evaluation formulas.
- No changes to few-shot momentum until complete 500-epoch evidence supports such a change.
- No Pillow `FreeTypeFont.getsize` compatibility fix in this patch; that issue is non-blocking and will remain an independent change.
- No weight, dataset, cache, or run-directory files will be committed.

## Verification

Implementation is accepted only if:

1. A new profile test fails before the Base file exists or while it has the wrong momentum.
2. The focused profile tests pass after the change.
3. The complete unit-test suite passes.
4. The relevant YAML files parse to the expected optimizer values.
5. Shell syntax checks for the existing launch scripts still pass.
6. `git diff --check` passes and the upstream-difference ledger covers every changed path.

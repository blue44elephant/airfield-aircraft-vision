# AW-01B aircraft detector diagnosis and improvement

Date: 2026-10-21. Repository: blue44elephant/airfield-aircraft-vision.
Branch main, HEAD a857820. AW-01 changes were already uncommitted and were preserved in an ignored source snapshot
before this work. No branch switch, commit, push, merge or external-project integration occurred.

## Outcome

**The detector improved, but reliable generalization remains unresolved. It is not operationally ready.**

On the new confirmatory test cohort, recall increased from 0% to 21.0% and count MAE decreased from 2.79 to 1.93 per tile.
The model still missed 64 of 81 labels and produced 18 false positives. The larger improvement on the old test cohort
must not be substituted for the weaker fresh-test result.

Data/model access was available. All reported measurements came from actual CPU inference and training, not estimates.
The original AW-01 checkpoint, manifests and metric reports remain unchanged.

## 1. Diagnosis before broader training

### Parsing, pixels and preprocessing

The initial audit checked cached training/validation imagery against its original COCO records, then the expanded
audit checked **220 training/validation images and 723 labels**.

- COCO XYWH boxes were correctly converted to clipped XYXY source-pixel edges.
- Native raster dimensions and RGB ordering matched the images. No channel swap was found.
- Input values are uint8 RGB divided by 255, followed once by ImageNet mean/std normalization.
- Model resizing is disabled in training and inference. Padding preserves the source grid.
- Tiling uses native 512-pixel windows with 128-pixel overlap; source affine/CRS and scene-wide NMS remain intact.
- RPN NMS is 0.7, ROI NMS 0.5 and scene NMS 0.4. None was selected using test labels.
- Model-level confidence floor is explicit: 0.05 for legacy defaults, 0.005 for diagnostic/AP collection.
- Flips/90-degree rotations added for training transform pixels and boxes together; all eight variants have regressions.

No coordinate, channel-order or double-normalization error was demonstrated.
[Detailed audit](aw01b/annotation_audit.json), [diagnosis evidence](aw01b/diagnosis.json).

### Undertraining and coverage

The original checkpoint detected only **9/61 original training labels at confidence 0.5**.
Thus its failure was already present on training data. Reducing the validation threshold recovered aircraft but caused
many false positives: on the original six validation tiles, threshold 0.2 gave 9 TP/18 FP/4 FN versus 3 TP/2 FP/10 FN at 0.5.
Threshold selection alone was insufficient.

The original training set had nine tiles, three groups and three captures. Its box-size median was 28.6 pixels,
with no box exceeding 38.8 pixels in square-root area. The exposed legacy test median was 65.4 pixels and reached 200.7.
The original data also concentrated 48/61 labels in one airfield group.

### Small-sample overfit

The unchanged original training path was tested on two known training images containing eight labels, with the
backbone frozen and original SGD settings. At step 0: 4 TP, 2 FP, 4 FN at confidence 0.5.
After 80 updates: **8 TP, 0 FP, 0 FN; precision/recall 100%; count MAE 0**.
This verifies that the pipeline can learn those known examples; it is not a generalization result.

### Architectural and annotation limits

The actual MobileNet feature maps for a 512-pixel image were 16 x 16, 16 x 16 and 8 x 8: strides 32/32/64.
Very small visible boxes may therefore occupy about one feature cell. This is a model limitation, not raster upscaling.

Visual label audits found both tiny edge fragments and heavily truncated large aircraft.
The aircraft-only instances files lack visible-fraction fields, so the implementation does not invent a truncation fraction.
Source-pixel box-size bins below are visibility-dependent proxies, not true aircraft length/wingspan classes.
No operational aircraft-type classification was introduced.

The evidence supports a combination of undertraining, narrow data coverage, threshold/calibration issues and small-object/
truncation difficulty. It does not quantify each factor's causal contribution.

## 2. Expanded dataset and independence

| Dataset / split | Native tiles | Airfield groups | Aircraft annotations |
| --- | ---: | ---: | ---: |
| AW-01 training | 9 | 3 | 61 |
| AW-01 validation | 6 | 2 | 13 |
| Expanded training | 174 | 16 | 576 |
| Expanded validation | 46 | 6 | 147 |
| Fresh confirmatory test | 29 | 4 | 81 |
| Legacy regression test | 9 | 3 | 28 |

Training covers 13 countries and 40 capture identifiers. It includes 164 clear-sky tiles, nine with cloud/haze metadata
and one with snow metadata. Training box square-root areas span 14.4?223.4 pixels.

The first expansion's 30-tile validation set still contained only small boxes. Before test evaluation, two unused
official-training groups were added to validation for larger aircraft and cloud/haze conditions.
The resulting 46-tile validation set spans 14.8?204.5 pixels and includes eight cloud/haze tiles.
The early pilot checkpoint/history was retained but excluded from final selection.
Training and test samples were identical before/after this validation-only revision.

Fresh test locations: Flugplatz Antersberg (Germany), Hagerstown Regional, Salt Lake City International and
Killeen Municipal (USA). These four groups are absent from AW-01 and expanded training/validation.
The old test cohort is retained separately as a regression cohort because it was already exposed during AW-01 diagnosis.

Split protection:
- preserve old validation/test groups outside training;
- union shared location IDs, capture IDs, country/airfield names and geographic centres within 10 km;
- prohibit cross-split capture/hash reuse;
- reject selected footprint-envelope overlaps and distances below 1 km;
- observed minimum cross-split footprint-envelope distance: **92.8 km**;
- select fresh test tiles by seeded filename order, without using their detection labels to choose the model.

Native downloaded bytes, sidecars, source URLs, CRS/affine, separate source-PAN resolution/output spacing, acquisition
strings and minute precision are retained. UTC basis remains explicitly assumed, not newly verified.
Original source imagery is never resampled or overwritten.

The initial expansion used a 128 MiB network bound/four workers, followed by a separate 16 MiB validation bound.
No full scenes, recursive bucket download or credentials were used.
[Manifest](aw01b/dataset_manifest.json), [frozen protocol](aw01b/protocol.json).

RarePlanes-derived manifests and images retain attribution and CC BY-SA 4.0; see
[attribution](aw01b/ATTRIBUTION.md) and the [existing dataset assessment](AW01_DATASET.md).

## 3. Training and validation selection

Architecture retained: TorchVision Faster R-CNN MobileNetV3-Large FPN.
Both candidates start from the same verified official COCO airplane initializer.
Native input, normalization and box coordinates are preserved; no sharpening/upscaling/super-resolution is used.

| Setting | Expanded frozen | Expanded trainable FPN |
| --- | --- | --- |
| Body | frozen | frozen |
| Feature pyramid | frozen | trainable |
| Optimizer | SGD, LR 0.005, momentum 0.9, WD 0.0005 | AdamW, LR 0.0003, WD 0.0001 |
| Epoch budget | 6 | 6 |
| Batch / seed / CPU threads | 2 / 23 / 4 | 2 / 23 / 4 |
| Augmentation | native D4 flips/90? rotations | same |
| Gradient norm limit | 10 | 10 |

Normalization statistics remain frozen. Each epoch is evaluated on validation only.
Threshold grid: 0.05/0.10/0.15/0.20/0.25/0.30/0.40/0.50/0.60/0.70/0.80.
Rank by validation F1, then lower count MAE, then higher threshold. AP50 is reported but is not the selection target.
These are two complete training profiles, not a clean one-factor ablation.

| Candidate | Selected epoch | Threshold | Validation P / R / F1 | AP50 | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| expanded_frozen_sgd | 3 | 0.4 | 70.7% / 39.5% / 50.7% | 43.7% | 1.761 |
| expanded_fpn_adamw | 3 | 0.3 | 49.5% / 34.0% / 40.3% | 34.3% | 2.087 |

**Selected: expanded frozen, epoch 3, confidence 0.4.** The worse FPN run was not promoted.
The original AW-01 operating point selected on the same validation set was 0.2.
A selection-lock JSON was written before independent evaluation; no threshold/model tuning followed the test results.

Selected checkpoint: artifacts/aw01b/balanced_frozen_sgd/best.pt.
SHA-256: b455a4c91321e6c18ec2ea01995ad897356390d881a06aecf10242b0cdb10d70.
Original checkpoint SHA-256 remains:
3f96b9ff3b42ebbc937eb818f2582de3e087e9eee654571363e778b8257b70fc.

[Selection lock](aw01b/selection-lock.json), [training histories](aw01b/training_histories.json).

## 4. Independent evaluation

Precision/recall/F1 use confidence-ordered one-to-one matching at IoU 0.5.
AP50 is 101-point interpolated AP at IoU 0.5 over candidates down to score 0.005, not multi-IoU COCO AP.
AP is threshold-independent here, so the two AW-01 operating-point rows have the same AP50.
Undefined precision is JSON null when there are no accepted predictions.

### Fresh confirmatory cohort: 29 tiles, 81 labels, four groups

| Model / operating point | Precision | Recall | F1 | AP50 | Count MAE | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-detection baseline | undefined | 0.0% | 0.0% | 0.0% | 2.793 | 0 / 0 / 81 |
| AW-01, confidence 0.5 | undefined | 0.0% | 0.0% | 9.1% | 2.793 | 0 / 0 / 81 |
| AW-01, validation-selected 0.2 | 37.5% | 3.7% | 6.7% | 9.1% | 2.655 | 3 / 5 / 78 |
| AW-01B, validation-selected 0.4 | 48.6% | 21.0% | 29.3% | 25.7% | 1.931 | 17 / 18 / 64 |

The count MAE reduction versus both original confidence 0.5 and the zero-count baseline is 30.9%.
Nevertheless, 79.0% of labels were missed and 51.4% of accepted detections were unmatched at IoU 0.5.
Count agreement can hide cancelling false positives/negatives.

### Legacy regression cohort: nine tiles, 28 labels

| Model / operating point | Precision | Recall | F1 | AP50 | Count MAE | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-detection baseline | undefined | 0.0% | 0.0% | 0.0% | 3.111 | 0 / 0 / 28 |
| AW-01, confidence 0.5 | undefined | 0.0% | 0.0% | 10.4% | 3.111 | 0 / 0 / 28 |
| AW-01, validation-selected 0.2 | 27.3% | 10.7% | 15.4% | 10.4% | 1.889 | 3 / 8 / 25 |
| AW-01B, validation-selected 0.4 | 73.9% | 60.7% | 66.7% | 65.2% | 1.000 | 17 / 6 / 11 |

This cohort is reported for continuity with AW-01, not as the sole independent improvement claim.
The original 0.5 result was reproduced exactly.

### Fresh test by unseen group

| Group | Tiles | Labels | Precision | Recall | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Antersberg (Germany) | 7 | 20 | 60.0% | 15.0% | 2.143 |
| Hagerstown (USA) | 6 | 27 | 50.0% | 7.4% | 3.833 |
| Salt Lake City (USA) | 8 | 14 | 46.2% | 42.9% | 0.875 |
| Killeen (USA) | 8 | 20 | 46.2% | 30.0% | 1.375 |

### Fresh test by source-pixel box size

| Box-area bin | Labels | Matched labels | Recall | Per-tile count MAE |
| --- | ---: | ---: | ---: | ---: |
| small_lt32px | 43 | 6 | 14.0% | 1.414 |
| medium_32to96px | 36 | 9 | 25.0% | 1.000 |
| large_ge96px | 2 | 2 | 100.0% | 0.069 |

Bins use bounding-box area: below 32? pixels, 32??96?, and at least 96?.
These are not aircraft-type or verified physical-wingspan categories. Truncation, orientation and pixel spacing affect them.
Predicted-size count error can include localization crossing a bin boundary; per-bin MAEs do not add to the overall MAE.
Only two large labels were present, so their 100% recall is not a reliable large-aircraft performance estimate.

### Fresh test by dataset weather

| Dataset weather | Tiles | Labels | TP / FP / FN | Recall | Count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clear Skies | 24 | 73 | 15 / 14 / 58 | 20.5% | 2.083 |
| Snow | 3 | 4 | 0 / 2 / 4 | 0.0% | 0.667 |
| Cloud Cover or Haze | 2 | 4 | 2 / 2 / 2 | 50.0% | 2.000 |

Weather is coarse source-scene metadata, not measured AOI cloud/quality.
AOI quality remains unknown and was explicitly permitted for this experiment.
The snow/cloud strata are tiny; no broad weather-performance conclusion is justified.

[Complete comparisons, strata and per-image FP/FN](aw01b/comparison.json).

## 5. Representative successes and failures

[Reviewed native-size gallery](examples/aw01b/README.md):
- fresh small-aircraft success: one label, one matched detection;
- fresh snow/background/localization failure: one TP, three FP, one FN;
- fresh crowded-apron failure: ten labels, no accepted detections;
- legacy large-aircraft improvement: two matched detections on the exact previously downloaded scene.

The snow example includes a high-score box on an aircraft that fails the IoU threshold because of localization,
plus additional unmatched patterns. FP/FN terminology is relative to supplied labels; not every FP is a different physical object.
No visual examples are substituted for the full aggregate metrics.
All 38 test overlays and result JSON files are retained under ignored artifacts/aw01b/comparison.

## 6. Software changes and verification

- Reused the existing router, schemas, native raster reader, NMS, detector builder and result contracts.
- Added bounded geographically grouped preparation, native augmentation, validation checkpoint/threshold selection,
  AP50/F1/strata, locked independent comparison and reproducible AW-01B commands.
- Existing inference now honors a checkpoint's validation threshold by default; old checkpoints retain the 0.5 fallback.
- Fixed parallel download budget reservation and rejected truncated/mismatched Content-Length responses before installation.
- Fixed silent reuse of a different requested dataset configuration.
- Fixed unnecessary writes of an identical frozen manifest. Original preparation now returns cached data without rewriting it.
- Added regressions for geometry/augmentation, split leakage, selection restrictions, metrics, download integrity,
  missing prediction records, inference defaults and frozen-cache behavior.

**Complete automated suite: 93 passed** (66 existing, 27 added cases).
Actual CPU diagnosis, both six-epoch runs, fresh/legacy inference and result exports completed.
Installed CLI/module entry points and the original cached preparation workflow were exercised.
No routine test downloads models or imagery.

The project-local venv uses official NumPy 2.4.2 instead of the experimental inherited NumPy 1.26.4.
Global Python and protected project environments were not changed.
[Environment](aw01b/environment.json), [exact Windows commands](AW01B_WORKFLOWS.md).

Source-image/sidecar hashes and native overlay dimensions were checked.
Original checkpoint, manifest and historical AW-01 reports were preserved.
Git diff/whitespace review passed; branch and HEAD are unchanged.
[AW-01B file inventory](AW01B_CHANGED_FILES.md).

## 7. Remaining limitations and next experiment

This milestone establishes an improved experimental detector, not reliable aircraft inventory.
Fresh-test small-box recall is only 14.0%; crowded scenes and localization errors remain serious problems.
Training/validation are still limited, positive-only, source-correlated samples from one dataset/sensor.
Confidence is uncalibrated. Empty-airfield false-alarm rates, cross-sensor transfer, true visibility fractions and
independent AOI quality categories remain unverified.

Recommended next experiment:
1. Freeze a new untouched test cohort; retain this fresh cohort as regression data.
2. Use richer visibility annotations/native source context to handle heavily truncated aircraft.
   Do not silently relabel visible fragments as verified empty background.
3. Add verified empty aprons/hard negatives and stronger crowded/small-aircraft validation.
4. Compare the current stride-32 features with a bounded pretrained finer-feature detector, such as
   [TorchVision ResNet-50 FPN](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html),
   keeping the same native-pixel/provenance contracts. This is a proposed comparison, not an implemented replacement.
5. Evaluate learning-rate schedules and confidence calibration on validation, then repeat independent tests once.

No Blue Elephant integration, SAR aircraft detection, UAV-specific classification, ship-detector change or tracking change
was implemented. No automatic commit, push, merge or branch switch occurred.

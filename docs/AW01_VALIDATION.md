# AW-01 measured validation

Date: 2026-10-21. Repository: `blue44elephant/airfield-aircraft-vision`.
Branch `main`, HEAD `a857820`; initially clean. Changes are uncommitted.
No branch switch, commit, push or merge. Blue Elephant, Epic Archer and Sentinel Ship Motion were not modified.

## Outcome

**The trainable optical pipeline works, but the measured checkpoint fails the unseen-airfield test.**
It must not be used as an operational aircraft counter. This is a negative baseline result, not successful
generalization or validated inventory estimation.

Data and weights were accessible: anonymous RarePlanes metadata, selected native PNG tiles with XML sidecars,
and the official TorchVision COCO initializer were actually downloaded. Training, inference and evaluation ran on CPU.
There is no dataset/weight-access blocker. Accuracy, small sample size and the nonstandard local numerical environment
are the material limitations.

## Implemented functionality

- Existing router and VISION-0 Pydantic contracts reused unchanged.
- High-resolution optical-only native-pixel tiled inference; scene-wide NMS; boxes, uncalibrated scores, accepted/rejected
  candidates and visible-detection counts.
- Source-affine WGS84 polygons when georeferencing is supplied. No image warp, resize, sharpening or artificial resolution gain.
- Explicit uint8 RGB validation, source/sidecar checksums and pixel-grid consistency checks.
- Source image and band/panchromatic resolution evidence, separate X/Y output spacing, acquisition precision and timezone basis.
- Unknown-quality override is explicit; nodata and unknown UTC basis produce warnings. All-nodata fails instead of reporting zero.
- Bounded metadata/data preparation; deterministic split selection; no airfield/capture/hash leakage across train/validation/test.
- Frozen-backbone TorchVision fine-tuning; checkpoint lineage; independent-airfield metric evaluation and native-size overlays.
- Standalone `aircraft-vision` / `python -m airfield_aircraft_vision` CLI.

No provider credentials, SAR inference, aircraft-type/UAV-model classification, heading estimation or operational integration
was added. Original source imagery is untouched.

## Experiment

Seed 17; native 512 x 512 tiles; 512-pixel inference windows; 128-pixel overlap; confidence 0.5; scene NMS IoU 0.4.
IoU matching threshold 0.5, confidence-ordered one-to-one matching. No test-set parameter tuning.

| Split | Tiles | Independent groups | Aircraft labels |
| --- | ---: | ---: | ---: |
| Train | 9 | 3 | 61 |
| Validation | 6 | 2 | 13 |
| Test | 9 | 3 | 28 |

Training: Buchanan Field (USA), Cordoba (Spain), Capital City (USA).
Validation: Tatenhill (UK), Manassas (USA).
Test: Shanghai Pudong (PRC), Copenhagen (Denmark), Albany and Schenectady County (USA).
The latter two locations belong to one unioned capture/location group, so there are four named test airfields
in three independent groups. Test groups are absent from the entire official-training group set, not just our nine tiles.
This establishes separation from AW-01 fine-tuning; COCO pretraining locations cannot be fully audited here.

The source index contains 67 official-training groups and 16 test-only groups after lineage union; 17 groups occur
in both official partitions. The supplied observation-based split is therefore not sufficient by itself.

Model: Faster R-CNN MobileNetV3-Large FPN, official COCO airplane initialization, binary head, frozen backbone including
normalization statistics. SGD, learning rate 0.005, momentum 0.9, weight decay 0.0005; three fixed epochs; four CPU threads.
Mean training losses: 0.528519, 0.446524, 0.524262. No convergence claim.
[Model API](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html).

## Measured results

| Metric | Validation | Independent test |
| --- | ---: | ---: |
| True positives | 3 | 0 |
| False positives | 2 | 0 |
| False negatives | 10 | 28 |
| Precision | 60.00% | undefined (no predictions) |
| Recall | 23.08% | 0.00% |
| Accepted detections | 5 | 0 |
| Count MAE per tile | 1.3333 | 3.1111 |
| Count bias per tile | -1.3333 | -3.1111 |
| Count RMSE per tile | 1.9149 | 4.0552 |
| Sum of absolute count errors | 8 | 28 |

These are micro precision/recall and per-tile count errors, not mAP or whole-airfield inventory.
Undefined precision is stored as JSON null; it is not perfect precision.

[Validation metrics](aw01/validation_metrics.json), [test metrics](aw01/test_metrics.json),
[frozen experiment manifest](aw01/experiment_manifest.json), [environment](aw01/environment.json).
The manifest is an archival byte-for-byte copy; its image paths resolve beneath the original `data/rareplanes` root.
Full per-image detections/overlays are in ignored `artifacts/aw01-validation-final` and `artifacts/aw01-test-final`.
The checkpoint is `artifacts/aw01-3epoch.pt` (about 73 MiB; not committed).

Checkpoint SHA-256:
`3f96b9ff3b42ebbc937eb818f2582de3e087e9eee654571363e778b8257b70fc`.
Training manifest SHA-256:
`311cb70ee56a3765b86f539f5a075fb8deb4ad44ae14b3d04b4098c92f029138`.

## Visual review

[Native-size examples and captions](examples/aw01/README.md) include:
- Tatenhill tile 115: one matched validation detection.
- Manassas tile 46: a matched aircraft, an unmatched partial duplicate around that aircraft, and a missed aircraft.
  Its predicted count equals the labelled count despite the FP and FN: count agreement alone conceals detection mistakes.
- Schenectady tile 102: ten labelled aircraft missed on the independent test set.

Examples were visually inspected. Cyan denotes truth/FN, green matched predictions, red unmatched predictions.
All 15 evaluation overlays remain 512 x 512. Original image hashes and sidecar hashes were checked after inference.
Representative images are derivatives under CC BY-SA 4.0 with attribution, not proprietary-code assets.

## Provenance findings

Source panchromatic GSD across the sample is about 0.317?0.393 m. Local delivered X/Y spacing spans about
0.179?0.379 m, depending on latitude/grid direction. Finer grid spacing is not claimed as finer optical resolution.
Source affine grids are EPSG:4326; metric spacing is calculated by transforming points only.
The original RGB is never resampled. A real TorchVision-transform regression confirms spatial dimensions and box coordinates
remain unchanged through model normalization/padding.

CSV collection times have minute precision. Raw strings and an explicit **assumed UTC** basis are retained.
No exact seconds or provider-confirmed UTC basis were established. The old exact-time transport result is withheld
for these samples; the richer experimental result retains the sensing-time evidence and limitation.
Local cloud/occlusion quality is unknown and explicitly allowed only for the experiment.

## Confirmed implementation issues addressed

1. Preparation initially assumed a projected metric CRS; the live EPSG:4326 sidecars disproved this.
   Added local X/Y metre calculation and a geographic-grid regression.
2. Selected training groups present in both official partitions could contribute tiles twice, exceeding the configured bound.
   Selection now uses only the proper official partition per assigned group; deterministic regression verifies the cap.
   The preliminary manifest was preserved as `data/rareplanes/manifest.before-group-bound-fix.json`; source imagery retained.
3. Entirely nodata scenes could otherwise resemble zero aircraft. Such inputs now fail explicitly.
4. Sidecar changes, mismatched source grids, ambiguous RGB channels and invalid spatial metadata are rejected.
5. Assumed/unknown UTC evidence is explicit in inference and metric records; no silent promotion into exact-time export.
6. Test runner initially lacked its parent output directory; documented commands create `artifacts` first.

## Verification

- Baseline tests: **30 passed**.
- Final complete suite: **66 passed** (36 added cases), JUnit at `artifacts/aw01-tests.xml`.
- Actual CPU fine-tuning, six-tile validation and nine-tile test evaluation completed.
- Final evaluation repeated after RGB/provenance guards; aggregate metrics remained identical.
- Local `infer` command returned one accepted aircraft on Tatenhill tile 115.
- Isolated editable install in local `.venv` succeeded; installed `aircraft-vision --help` succeeded.
  This verification venv inherits existing system packages; fresh-environment commands are documented separately.
- Native originals, georeferencing checksums and overlay dimensions checked.
- `git diff --check` and new-text whitespace checks passed; branch and HEAD unchanged.
- No GUI existed to exercise. This milestone verifies CLI/library behavior, not a desktop integration.

## Remaining limitations and next experiment

The three-epoch, nine-tile run is far too small to establish a reliable detector. It demonstrated poor transfer,
including zero test recall. Limited training coverage, scale/background/domain differences and frozen features are plausible
contributors; no ablation establishes their relative cause. Do not lower the threshold against this frozen test set
to manufacture a better headline result.

Next work should use broader training/validation airfields and negatives, a validation-led training/model sweep,
and a new untouched test cohort. Include crowded/large/truncated aircraft, empty aprons, clouds/shadows, calibration,
geographic-distance leakage checks and image-level false-alarm metrics. Partial/fragment duplicates can survive IoU NMS.

The current CPU environment uses Python 3.14.3 and experimental NumPy 1.26.4 with runtime warnings.
A clean supported environment, larger training and independent revalidation are prerequisites for deployment.
No claim is made for SAR, medium/low-resolution imagery, UAV types, global availability or operational inventory.

See [exact Windows commands](AW01_WORKFLOWS.md), [data terms](AW01_DATASET.md),
[integration recommendations](AW01_BLUE_ELEPHANT_INTEGRATION.md) and [changed-file inventory](AW01_CHANGED_FILES.md).

# AW-01C - Aircraft detector generalization validation

Work dates: 2026-10-21 to 2026-10-22 (Asia/Calcutta). Repository: `C:\Tools\Inventions\airfield-aircraft-vision`.
Starting branch: `feat/aw-01c-generalization`; HEAD: `137cdf6` (`Minor tweaks and training`).
The working tree was clean before implementation. Original AW-01/AW-01B experiments were preserved.

## Evidence and diagnosis

The six required AW-01B documents/JSON records were reviewed and their hashes recorded in [failure_review.json](aw01c/failure_review.json). Prior coordinate, RGB-channel, normalization and 80-step overfitting checks succeeded (8/8 known aircraft learned). No evidence supports a coordinate or channel-order explanation for the remaining failures.

- AW-01B's MobileNet feature maps at native 512 pixels have strides **32/32/64**. Aircraft with small visible boxes are poorly represented on that coarse grid.
- On its exposed fresh test, small-box recall was 14%; three crowded tiles contained 25 labels but only three true positives (12% recall, count MAE 7).
- Unmatched overlapping boxes include localization errors; an unmatched detection is not proof that the depicted object is not an aircraft.
- The frozen MobileNet run peaked on validation at epoch three. Later validation F1 declined; the FPN AdamW trial also worsened. Simply adding epochs at the same learning rate was not justified.
- Training was limited and all original tiles were positive. Empty-scene false alarms had never been measured.

[annotation_review.json](aw01c/annotation_review.json) joins the richer official annotations by image/annotation ID and exact box. All 18,393 training-index annotation IDs/boxes match the simplified index. The expanded training cohort contains 23 labels with visible fraction below 0.25 (1.07%). **None of its 1,307 small-box labels is such a severely clipped fragment**; those 23 fragments are in the medium box-area bin. Fragment noise exists but does not explain most small-aircraft misses. All source labels are retained; a visibility-exclusion experiment remains separate future work.

These observations motivate the experiment; they do not establish that a single cause explains every miss.

## Dataset, separation and provenance

| Cohort | Aircraft tiles | Reviewed background crops | Aircraft labels | Airfield groups |
|---|---:|---:|---:|---:|
| AW-01B training | 174 | 0 | 576 | 16 |
| AW-01C training | 452 | 15 | 2,150 | 32 |
| AW-01C validation | 46 | 6 | 147 | 6 |
| New independent test | 68 | 0 | 228 | 9 |
| Exposed AW-01B regression | 29 | 0 | 81 | 4 |

Training covers 68 capture groups in 13 countries: Albania, Bahamas, Belgium, Croatia, Denmark, France, Germany, India, Italy, Portugal, Spain, Switzerland and USA. Independent testing uses 14 capture groups in Bahamas, France, Germany, Italy and USA. These are unseen airfields/captures within the same RarePlanes/WorldView-3 domain, not proof of transfer to other sensors or all regions.

The [test reservation](aw01c/test-reservation.json) fixed all nine remaining official-test-only unexposed geographic groups and 68 seed-selected image IDs **before model experiments**. Original and previously exposed test airfields were excluded from training/validation. Original AW-01B train/validation records remain equivalent in the isolated expansion. The prior AW-01B fresh test is regression only.

The existing geographic grouping and shared-capture exclusion are reused. Selected cross-split footprint envelopes do not overlap; the minimum envelope distance is **42.3258 km**. Tile overlaps can still occur within a cohort, so per-tile counts cannot be summed into airport inventories.

Training additions alternate small-box, crowded and seeded examples with capture round robin. Holdout labels do not rank training tiles. See [protocol](aw01c/protocol.json) and [frozen manifest](aw01c/dataset_manifest.json).

### Native pixels and quality checks

[pixel_audit.json](aw01c/pixel_audit.json) checks all 519 training/validation inputs:

- Native dimensions, uint8 RGB, CRS and affine match their source records.
- Rasterio and Pillow decode identical RGB pixels for every input.
- All 2,150 training and 147 validation label centres fall on valid pixels.
- No entirely invalid image or ground-truth region was found.
- Thirteen training and two validation inputs have partial no-data borders; the existing mask policy remains in use. Seven training boxes include some invalid surrounding pixels but have valid centres.
- All 21 accepted negative crops are fully valid. One initially reviewed candidate was rejected for only 38.28% valid pixels.

Native physical resolution metadata spans **0.311046-0.392829 m**. Delivered geographic pixel spacing remains separately recorded. There is no resizing, reprojection, sharpening or synthetic resolution increase. Existing padding and ImageNet normalization are retained.

### Reviewed negative crops

The public tiled imagery is positive-only. A deterministic proposal step excludes every original/ignored box plus a 32-pixel halo, then requires an explicit parent-checksum/window-bound visual review.

Twenty-one native 256-pixel crops were approved by **Codex visual review**, not by a separate human survey: training has two empty-apron and 13 difficult-background crops; validation has one empty-apron and five difficult-background crops. Roads, vegetation and buildings are classified as backgrounds, not aprons. [Review records](aw01c/negative_reviews.json) identify each exact source and decision.

Lossless derived GeoTIFFs preserve native pixels, adjusted affine, CRS, source timestamp, physical resolution, grouping, licensing and parent/review hashes. Original imagery/sidecars remain unchanged. All approvals and inputs are validated before output mutation; exact replay verifies existing bytes.

**The independent test contains no verified negatives.** Validation background findings do not establish an independent full-scene false-alarm rate.

### Access and usage conditions

RarePlanes real imagery/annotations are publicly accessible under CC BY-SA 4.0. Attribution and derivative changes are recorded in [ATTRIBUTION.md](aw01c/ATTRIBUTION.md). The [owner's guide](https://www.iqt.org/library/rareplanes-public-user-guide) documents the native positive tiles and visibility fields; the [dataset page](https://www.iqt.org/library/the-rareplanes-dataset) documents its scope and licensing.

New imagery acquisition was limited to **118,146,582 bytes** against a 192 MiB budget, four workers and per-file limits. The richer training metadata used 33,539,381 bytes under its separate 40 MiB bound. No full scenes, synthetic imagery, Sentinel data or SAR imagery were downloaded.

## Architecture feasibility and controlled training

The original MobileNet baseline/checkpoint remains supported. The alternative is TorchVision **Faster R-CNN ResNet-50 FPN**, initialized from the single official COCO_V1 artifact (167,502,836 bytes; SHA256 `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`). The exact FrozenBatchNorm graph is reconstructed and background/airplane head rows transferred. Model constructors do not implicitly download weights.

Official references: [TorchVision 0.25 detector/input contract](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html), [BSD code license](https://github.com/pytorch/vision/blob/v0.25.0/LICENSE). Pretrained-data usage conditions are separate from code licensing; weights remain local.

A bounded random-weight shape/cost probe preceded pretrained acquisition. Native 512 ResNet FPN strides are **4/8/16/32/64**. The pretrained feasibility probe measured 1.125 s inference, 0.969 s bounded frozen-backbone training step, and peak working set about 702 MiB. Random weights were used only for feasibility, never as the final initializer. See [shape/cost probe](aw01c/architecture-feasibility.json) and [pretrained probe](aw01c/pretrained-feasibility.json).

Both new training arms share:

- Identical frozen data, seed 41, batch one and four CPU threads.
- Entire backbone and FPN frozen; RPN and ROI heads trained from compatible COCO initialization.
- Native D4 flips/90-degree rotations, no pixel scaling; seeded airfield round robin visits every sample once per complete epoch.
- SGD, learning rate 0.0025, momentum 0.9, weight decay 0.0005, gradient clip 10.
- Learning rate multiplied by 0.3 after each epoch.
- Training proposal limits 1,000/512 and 128 ROI samples; inference proposal defaults retained.
- At most three epochs; update budgets 900 s MobileNet / 1,500 s ResNet, followed by final validation/checkpoint bookkeeping.
- Validation-only F1 selection, then lower count MAE and higher threshold; fixed inherited threshold grid and score floor 0.005.

This compares two architectures under the same new preparation/training protocol. Changes from AW-01B also include data, sampling and schedule; the improvement relative to AW-01B cannot be attributed exclusively to architecture.

The reusable training command rejects incompatible physical-resolution/quality routes through the original VISION-0 Tier A policy, ignored training regions, invalid configurations, nonfinite loss and holdout exposure in continued checkpoints. Every current training/validation input passes the original Tier A policy. Cooperative STOP files and time checks bound execution; partial epochs are identified honestly.

## Actual training and frozen selection

| Candidate | Completed training | Elapsed | Peak process memory | Selected validation P / R / F1 | AP50 | Count MAE | Threshold |
|---|---|---:|---:|---|---:|---:|---:|
| expanded-mobilenet | E1: 467, E2: 467, E3: 467 | 7.31 min | 722.8 MiB | 80.3% / 36.1% / 49.8% | 44.7% | 1.788 | 0.7 |
| resnet50-fpn | E1: 467, E2: 467, E3: 170 (partial) | 26.15 min | 870.6 MiB | 81.2% / 73.5% / 77.1% | 80.8% | 0.692 | 0.7 |

ResNet stopped at the predeclared time limit after two full epochs and **170/467** updates of epoch three. The selected checkpoint is that partial third epoch; it is not presented as three completed epochs. MobileNet completed all three. Total measured training time was about 33.46 minutes on the Ryzen 7 7730U CPU with four threads; no GPU was used.

ResNet loss fell from 0.791 to 0.660 over the two complete epochs. Its validation F1 rose 72.7% to 76.9% to 77.1%; AP50 rose 74.9% to 79.7% to 80.8%. The short partial epoch does not establish full convergence. MobileNet validation F1 rose 38.9% to 47.2% to 49.8%.

The selection rule chose ResNet and threshold **0.7** using validation only. [selection-lock.json](aw01c/selection-lock.json) was written before test parsing/prediction. Selected checkpoint SHA256:

~~~text
d2aecd3624648981f35def1fff7bb36789f9635546f4f848c719c48e647eda42
~~~

Each selected model and the preserved AW-01B checkpoint was inferred once per cohort. Examples reuse the saved predictions. No test result changed the model, threshold or training configuration.

Recorded source hashes identify the exact executed trainer; its local snapshot is retained at `artifacts/aw01c/experiment-code/generalization_training.py`. Later preflight guards and formatting do not alter the training path for these already valid inputs. Wall-clock limits can yield a different number of updates on another machine; exact metric replay uses the preserved checkpoint/prediction hashes rather than a claim of bit-identical timed retraining.

## Independent results - 68 tiles, 228 aircraft, nine unseen airfield groups

| Model | Precision | Recall | F1 | AP50 | Count MAE | Count bias | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---|
| Zero count | n/a | 0.0% | 0.0% | 0.0% | 3.353 | -3.353 | 0 / 0 / 228 |
| Preserved AW-01B, threshold 0.4 | 63.3% | 33.3% | 43.7% | 37.2% | 1.941 | -1.588 | 76 / 44 / 152 |
| AW-01C ResNet, threshold 0.7 | 81.2% | 75.9% | 78.5% | 84.6% | 1.044 | -0.221 | 173 / 40 / 55 |

**Matched-cohort count MAE decreased 46.2% versus AW-01B and 68.9% versus zero counting.** Recall improved by 42.5 percentage points. This comparison uses the same new test tiles for both checkpoints; the earlier AW-01B headline metrics came from a different cohort and are not substituted into this comparison.

AP50 is 101-point interpolated AP at IoU 0.5 with score floor 0.005, not multi-IoU COCO AP. Precision is undefined for a zero-prediction baseline. Confidence scores are not calibrated probabilities.

### Size and crowding

| Visible source-pixel box bin | Labels | AW-01B recall | AW-01C recall | AW-01B size-bin MAE | AW-01C size-bin MAE |
|---|---:|---:|---:|---:|---:|
| small_lt32px | 115 | 20.9% | 72.2% | 1.265 | 0.750 |
| medium_32to96px | 97 | 40.2% | 77.3% | 0.926 | 0.603 |
| large_ge96px | 16 | 81.2% | 93.8% | 0.103 | 0.044 |

Size is visible bounding-box area, not physical wingspan or aircraft type. Predicted-size count errors can include a localization crossing a bin boundary.

| Annotated aircraft per tile | Tiles / labels | AW-01B recall | AW-01C recall | AW-01B count MAE | AW-01C count MAE |
|---|---|---:|---:|---:|---:|
| one_labeled | 24 / 24 | 50.0% | 95.8% | 0.458 | 0.542 |
| six_or_more_labeled | 11 / 108 | 21.3% | 71.3% | 6.364 | 2.636 |
| two_to_five_labeled | 33 / 96 | 42.7% | 76.0% | 1.545 | 0.879 |

Small-box recall improved **24/115 to 83/115** (20.9% to 72.2%). Crowded-tile recall improved **23/108 to 77/108** (21.3% to 71.3%). Crowded scenes still have count MAE 2.636 and mean undercount 2.091 aircraft/tile. Single-aircraft tile MAE worsened from 0.458 to 0.542 because remaining false alarms cause overcounts; aggregate improvement is not universal.

### Airfields and geography

| Airfield group / metadata name | Tiles / labels | AW-01B recall | AW-01C recall | AW-01C MAE |
|---|---|---:|---:|---:|
| rareplanes-location-11: Flughafen Friedrichshafen, Germany | 8 / 14 | 35.7% | 78.6% | 0.750 |
| rareplanes-location-119: Portland-Mulino Airport, USA | 4 / 38 | 0.0% | 31.6% | 6.000 |
| rareplanes-location-124: Truckee-Tahoe Airport, USA | 8 / 41 | 14.6% | 73.2% | 1.125 |
| rareplanes-location-22: Perugia San Francesco d'Assisi - Umbria International Airport, Italy | 8 / 17 | 29.4% | 64.7% | 1.125 |
| rareplanes-location-5: Grand Bahama International Airport, Bahamas | 8 / 30 | 73.3% | 96.7% | 0.250 |
| rareplanes-location-58: Pocono Mountains Municipal Airport, USA | 8 / 34 | 26.5% | 97.1% | 0.250 |
| rareplanes-location-62: Naples Municipal Airport, USA | 8 / 30 | 40.0% | 86.7% | 0.625 |
| rareplanes-location-69: Tremblay-en-France Airport, France | 8 / 12 | 83.3% | 91.7% | 0.375 |
| rareplanes-location-93: Cincinnati/Northern Kentucky International Airport, USA | 8 / 12 | 58.3% | 83.3% | 1.375 |

| Country metadata | Tiles / labels | AW-01B recall | AW-01C recall | AW-01C MAE |
|---|---|---:|---:|---:|
| Bahamas | 8 / 30 | 73.3% | 96.7% | 0.250 |
| France | 8 / 12 | 83.3% | 91.7% | 0.375 |
| Germany | 8 / 14 | 35.7% | 78.6% | 0.750 |
| Italy | 8 / 17 | 29.4% | 64.7% | 1.125 |
| USA | 36 / 155 | 21.9% | 71.6% | 1.417 |

### Acquisition-condition proxy

| Dataset weather | Tiles / labels | AW-01B recall | AW-01C recall | AW-01C MAE |
|---|---|---:|---:|---:|
| Clear Skies | 60 / 198 | 32.3% | 74.2% | 1.100 |
| Cloud Cover or Haze | 8 / 30 | 40.0% | 86.7% | 0.625 |

These labels describe whole source scenes, not measured AOI-level cloud cover or an independently rated image-quality category. Actual per-image quality remains unknown. Small condition strata do not establish weather robustness.

### Reviewed validation backgrounds

| Validation category | Crops | Predictions | Count MAE |
|---|---:|---:|---:|
| verified_empty_apron | 1 | 0 | 0.000 |
| verified_background | 5 | 0 | 0.000 |

These six validation crops participated in selection. There are only **one apron and five other background validation crops**, no independent empty scenes, and no reliable deployment false-alarm estimate. The report leaves absent independent-negative statistics undefined.

## Exposed AW-01B regression - 29 tiles, 81 aircraft

| Model | Precision | Recall | F1 | AP50 | Count MAE | Bias | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|---|
| AW-01B | 48.6% | 21.0% | 29.3% | 25.7% | 1.931 | -1.586 | 17 / 18 / 64 |
| AW-01C | 70.0% | 69.1% | 69.6% | 77.9% | 1.069 | -0.034 | 56 / 24 / 25 |

The preserved baseline reproduces AW-01B's published 17 TP / 18 FP / 64 FN and AP50. This supports backward compatibility of the loader changes. Regression results are not independent confirmatory evidence; no checkpoint was selected from them.

## Detection examples and remaining errors

See [native comparison examples](examples/aw01c/README.md) and their [source/prediction index](examples/aw01c/index.json).

- **Small-aircraft improvement:** `124_104001001F2A5700_tile_126.png` improves from 0 TP / 1 FP / 12 FN to 10 TP / 2 FP / 2 FN.
- **Ground-equipment false alarms:** `93_1040010048202D00_tile_1555.png` detects both labeled aircraft but also returns four unmatched boxes around ground equipment/shadows.
- **Dense parking failure:** `119_104001003EA38900_tile_71.png` improves from zero to five matched aircraft, but still misses 12 of 17 labels; a merged/localized box is unmatched.

Unmatched boxes can include localization failures or possible label omissions. Some aircraft-like objects in the success image are unlabelled in the benchmark. No labels or scores were changed after observing the test output. Quantitative FP/FN remain defined against the frozen dataset labels.

## Verification, preservation and changed files

- **202 automated tests passed** (93 existing plus 109 new), with zero failures/skips. See [verification.json](aw01c/verification.json) and local `artifacts/aw01c/tests.xml`.
- New tests cover native feature grids/normalization, official weight/head compatibility, empty targets, deterministic sampling, bounds/cancellation, Tier A quality/physical-resolution gates, checkpoint leakage, rich annotation joins, metadata-cache integrity, exact native crop pixels/georeferencing/reviews, negative categories, fixed-threshold strata and pre-test selection locks.
- The final suite exposed a valid `other_background` review being omitted from the verified-background set. The mapping was corrected and regressed; no current experimental review used that category, so the frozen metrics are unaffected.
- Existing inference CLI loaded the selected ResNet checkpoint, used stored threshold 0.7, preserved native preprocessing, and produced one aircraft on a validation image. This separate startup smoke test did not re-infer any independent test image.
- `git diff --check` passed. Full modified/new-file inventory: [AW01C_CHANGED_FILES.md](AW01C_CHANGED_FILES.md).
- [Preservation verification](aw01c/preservation-after.json): all 28 original AW-01/AW-01B files match pre-experiment SHA256 and size; all 587 prepared images and 566 sidecars match their source records, zero mismatches.
- Branch and HEAD remain `feat/aw-01c-generalization` / `137cdf6`. No branch switch, commit, push or merge was performed. No Blue Elephant, Epic Archer or Sentinel Ship Motion files were modified.

Additional fixes harden the new workflows: cached preparation requires its request/protocol and metadata hashes; preserved train/validation records are checked exactly; matching protocols are not rewritten; continued checkpoints are checked against all current holdouts; invalid typed bounds, incompatible physical-resolution/quality routes and ignored-region training fail before expensive work.

## Remaining limits and next experiment

This is a promising research prototype for further Blue Elephant development, **not operational aircraft counting**. Forty unmatched predictions and 55 missed labels remain on the new cohort, with persistent undercounting in dense parking and occasional overcounting on isolated-aircraft tiles.

The nine airfield groups are a small, source-correlated sample from one dataset/sensor family. No confidence intervals are claimed; tiles within groups/captures can overlap, and geography/condition strata are small. Results do not establish cross-sensor, worldwide, full-scene or temporal generalization.

The backbone/FPN stayed frozen; longer controlled fine-tuning or small-object proposal changes remain untested. Native overlap/NMS behavior was retained rather than changed alongside architecture. Severe visible-fragment labels remain in training. Class scores are uncalibrated, and this milestone does not identify aircraft models or UAV classes.

The next experiment should reserve additional unexposed airfields and independently reviewed full-size empty aprons **before** tuning. Use training/validation-only ground-equipment hard-negative mining and one controlled small-object/localization change at a time; assess dense-apron misses and count errors on that new cohort. The current independent cohort is now exposed and must become regression data.

For future integration, keep the existing resolution/quality capability router and optical result/provenance contracts. Run cancellable inference in a separate worker with explicit experimental labeling and preserve native georeferenced detections. Ship detection, SAR inference and aircraft subtype/UAV classification remain outside this implementation. No integration was performed.

## Reproduction and evidence

[Exact Windows commands](AW01C_WORKFLOWS.md) cover preparation, bounded weights, visual-review replay, training, frozen comparison, testing and inference. [Environment versions](aw01c/environment.json), [training histories/resources](aw01c/training_histories.json), [comparison](aw01c/comparison.json) and [attribution](aw01c/ATTRIBUTION.md) accompany the report. Downloaded data, checkpoints and full action artifacts remain in ignored local `data/` and `artifacts/` directories.

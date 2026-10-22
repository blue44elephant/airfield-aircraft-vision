# AW-01D robustness and integration readiness

Validation date: 2026-10-22 Asia/Calcutta. Some host UTC evidence is dated 2026-10-21.

## Readiness decision

**A future analyst-supervised research adapter is technically supported. Automatic whole-airfield counting is not ready. No Blue Elephant integration was performed.**

Fresh tile accuracy is useful, but whole-scene background false alarms are substantial. Casa Grande produces 41 detections against 18 physical source labels; optional containment removes only one. New-sensor confidence calibration and independent positive-aircraft performance remain unverified.

## Preservation and audit

Branch **feat/aw-01c-generalization**, HEAD **7f71434**; clean starting tree. Actual checkpoint: artifacts/aw01c/resnet50-fpn/best.pt.

SHA-256: **d2aecd3624648981f35def1fff7bb36789f9635546f4f848c719c48e647eda42**, matching the AW-01C selection lock. [Before](aw01d/preservation-before.json) and [after](aw01d/preservation-after.json) verify 59 original AW-01/AW-01B/AW-01C checkpoint/metadata artifacts. No original checkpoint, image, annotation or sidecar was rewritten.

The architecture, native normalization, original inference paths and capability router are reused. Optional AW-01D modules add robust scene results, evaluation and bounded experiments. No other repository changes, Sentinel training, aircraft types, UAV classification, commit, push, merge or branch switch.

## Frozen data and independence

[Reservation](aw01d/robustness-reservation.json) preceded predictions. All 54 previously exposed groups were excluded. New sets: **48 positive evaluation tiles/eight groups/141 labels**, 24 training tiles/four groups, and 11 validation tiles/two groups. Four reviewed backgrounds were added separately to training and validation.

[Geographic audit](aw01d/geographic-independence.json): nearest prior-group annotation centroid **42.85 km**, no shared capture identifiers. Capture/name grouping plus a geographic audit support separation; centroid distances do not prove arbitrary full-scene footprints never intersect. Regions: USA, France and Germany.

Selection deliberately covers dense parking, small boxes, partial edge fragments and different captures. It is a diagnostic sample, not a random prevalence sample. Earlier AW-01C test labels are historical, exposed evidence and were not reused as a fresh test.

Additional, separately reported cohorts:

- 15 blind-reviewed native512 backgrounds from four evaluation captures, mostly runway, vegetation and buildings; not a comprehensive empty-apron survey.
- Two complete delivered source rasters: Gardner3390x3164 and CasaGrande5726x3588.
- Two fixed1536x1536 native ROIs: Montpellier and HotSprings, not complete airfields.
- Two independent1024x1024 NAIP aerial windows plus one reviewed512-pixel Cody background crop. Related captures/crops are never pooled into independent sample counts or inventories.

RarePlanes network retrieval: **48,485,823 bytes** =27,655,719 for83tiles +20,830,104 for6COGs. Requests were explicitly bounded, with no credentials or bulk retrieval. [Frozen manifests](aw01d/) retain hashes, original affine/CRS, parent windows and physical annotation IDs.

## Frozen AW-01C metrics

Unchanged operating point: raw acceptance **0.7**, candidate floor0.005, RGB512, overlap128, scene IoU NMS0.4. Review band0.2 to below0.7 is a triage category, not a count confidence interval.

| Cohort | Tiles | Precision | Recall | F1 | AP50 | Count MAE |
|---|---:|---:|---:|---:|---:|---:|
| Earlier AW-01C test, historical | 68 | 81.2% | 75.9% | 78.5% | 84.6% | 1.04 |
| Fresh AW-01D positive tiles | 48 | 80.3% | 80.9% | 80.6% | 85.0% | 0.60 |

Fresh: **114TP,28FP,27FN**;142predictions/141labels. Count bias+0.021, RMSE0.990. Different cohorts prevent interpreting this as a demonstrated model improvement. Safe inference reproduces baseline tile results exactly. Count agreement can conceal misses and false positives.

Matched-box mean IoU **0.779**, median0.787; median center error **2.61 native pixels**. Recall at IoU0.75 **53.9%**. Unmatched labels remain false negatives.

### Unseen geographic groups

| Group | Tiles | Precision | Recall | Count MAE |
|---|---:|---:|---:|---:|
| Montpellier | 6 | 77.3% | 77.3% | 0.33 |
| Hot Springs | 6 | 76.0% | 95.0% | 1.17 |
| Philadelphia | 6 | 70.6% | 75.0% | 0.50 |
| Augsburg | 6 | 87.5% | 77.8% | 0.33 |
| Seattle | 6 | 85.7% | 85.7% | 0.67 |
| Cleveland | 6 | 87.5% | 87.5% | 0.67 |
| Casa Grande | 6 | 87.5% | 77.8% | 0.67 |
| Gardner | 6 | 66.7% | 60.0% | 0.50 |

Crowded six-or-more-label stratum:8tiles/60labels, recall96.7%, precision86.6%, MAE0.88. Two-to-five labels: precision/recall70.5%; single-label recall65.0%. Few correlated groups do not establish regional reliability.

### Source-pixel object sizes

| Box category | Labels | Recall | Category count MAE/tile |
|---|---:|---:|---:|
| small_lt32px | 64 | 79.7% | 0.438 |
| medium_32to96px | 66 | 83.3% | 0.438 |
| large_ge96px | 11 | 72.7% | 0.104 |

Pixel-box bins are not aircraft types or physical size classes. Partial fragments are retained, but verified visibility-fraction labels are unavailable; no separate partial-aircraft recall claim is made.

### Metadata image conditions

| Condition | Tiles | Precision | Recall | Count MAE |
|---|---:|---:|---:|---:|
| Clear Skies | 37 | 82.7% | 78.6% | 0.51 |
| Snow | 7 | 78.6% | 91.7% | 1.14 |
| Cloud Cover or Haze | 4 | 68.8% | 78.6% | 0.50 |

Dataset weather is descriptive metadata, not an independently graded image-quality measure. Effective cloud occlusion, off-nadir severity and atmospheric variation are not comprehensively validated.

### Reviewed background false alarms

**One apparent FP in15windows:6.7% window false-alarm fraction,0.067FP/window.** Single Codex visual review is not independent expert adjudication. These texture-selected windows do not estimate full-airfield false-alarm prevalence.

The frozen negative manifest recorded its review checksum but omitted evaluator category-index fields, yielding an empty automatic reviewed-negative denominator. Overall FP records remain valid. [The supplement](aw01d/reviewed-negative-summary.json) reconstructs the denominator from frozen pre-prediction reviews and hashes. Future preparation preserves the category fields; locked manifests/reports were not rewritten.

## Whole-source and large-ROI counting

| Source/ROI | Native tiles | Labels | Default count | Containment count | Analyzable fraction |
|---|---:|---:|---:|---:|---:|
| 90_1040010003555D00.tif | 72 | 6 | 6 | 6 | 99.99748% |
| 49_1040010023B00E00.tif | 150 | 18 | 41 | 40 | 99.98006% |
| 10_1040010039679A00__1536.tif | 16 | 14 | 14 | 14 | 99.99797% |
| 101_104001003BD74200__1536.tif | 16 | 13 | 21 | 21 | 99.98635% |

All four are **partial** because small nodata regions exist. Traversal completed, but unavailable pixels do not imply aircraft absence. Counts describe accepted detections in analyzable pixels; true scene_total remains null. Entire delivered rasters do not guarantee entire-airfield coverage.

Complete-only primary metrics exclude these scenes. Explicit incomplete-source diagnostics against all physical source labels:

- Whole rasters:18TP/29FP/6FN, precision38.3%, recall75.0%, count MAE11.5.
- Large ROIs:23TP/12FP/4FN, precision65.7%, recall85.2%, count MAE4.0.

These are qualified diagnostics, not complete-coverage accuracy. Gardner's matching count alone does not establish correct localization.

### Duplicate suppression and native reconstruction

Original pixel offsets, overlap windows, source bounds and affine/CRS reconstruct coordinates without resizing, reprojection, sharpening or artificial detail. Global IoU NMS suppresses overlapping proposals. Optional boundary_containment additionally reconciles contained fragments from different internal cut edges, retaining the full box and an explicit heuristic flag. Default thresholds are unchanged.

CasaGrande default41 versus containment40 shows that deduplication alone does not fix the overcount. [Cached visual diagnosis](aw01d/scene-failure-evidence.json) attributes major errors to vehicle/equipment/building-ground confusers. Geometry-only rules cannot guarantee physical identity in crowded areas. No evaluation-label NMS tuning was performed, and perfectly deduplicated inventory is not claimed. Accurate unattended scene totals remain a blocker.

A new annotation-helper error was fixed **before inference**: clip geographic polygons before deriving window AABBs. AABB-only clipping added false corner intersections. [Alignment audit](aw01d/ground-truth-alignment.json): all325COCO boxes match reconstructed labels at IoU>=0.9999999996; geographic polygons additionally contribute19 genuine tiny fragments absent from simplified tiled COCO labels. Label sets are not identical. Original imagery/labels are unchanged; derived RGB, mask and affine roundtrips are exact.

## Independent aerial source

Two bounded NAIP windows: Moab2021-09-17 and Cody2022-07-06; native **0.6m**, EPSG:26912, exactRGB bands1-3 fromRGBIR. No resampling. Nearest RarePlanes annotations approximately293km/148km. RawSTAC16:00Z is retained as raw; only dates are verified.

[Official USDA terms](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) specify PublicDomain. The [collection](https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip) has a conflicting proprietary machine field/PublicDomain license link; both are recorded. Access uses the [official Rasterio opener contract](https://rasterio.readthedocs.io/en/stable/topics/vsi.html#python-file-and-filesystem-openers).

Completed retrieval41ranges/10.25MiB. An additional CodyGA-apron attempt stopped incomplete at20ranges/5MiB, produced no usable image and has no count; automatic retry is prohibited. No full parentCOGs downloaded. UnsignedURIs/ETags/rangeSHA/native-windowSHA preserve fetched-byte provenance; anonymous ephemeral access tokens are never stored.

**No verified positive-aircraft benchmark was obtained.** Moab's cross and a Cody edge object remain ignored/ambiguous. Moab has one accepted prediction outside its ignored matching region; Cody zero. The separately reviewed Cody background crop has zero accepted and four uncertain candidates. External positive recall is **undefined**. NAIP is aerial, not an independent satellite benchmark, and outside the validated sensor/GSD domain.

## Confidence, quality and failure handling

Calibration uses only the original52-image/six-group **validation** set. Raw accepted candidates108correct/133: Brier0.1322/ECE0.0829. Raw0.7-0.8 scores have empirical precision50% across28candidates.

Temperature0.5 minimizes all-candidate NLL but worsens accepted Brier to0.1634/ECE0.1594. **Not applied.** Group-held-out diagnostics are recorded; few folds do not establish transfer. References: [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html), [detection-specific calibration](https://arxiv.org/abs/2004.13546).

Correctness is conditional on a post-NMS candidate matched at IoU>=0.5. It does not estimate missed aircraft, count confidence, inventory completeness or spatial/size-aware calibration. Validation support is WV03/nativeGSD **0.317218-0.392725m**; wider experimental bounds do not establish empirical support.

New optional result semantics:

- Accepted, uncertain, rejected: raw scores, original pixel boxes, optional WGS84 polygons, flags/reasons.
- Complete/partial observed_count describes accepted detections only. Failed, unsuitable and cancelled counts are **null**, never zero aircraft.
- Scene_total always null; analyst verification always required.
- Exact pixel-union coverage avoids overlap double counting; unavailable tile windows are explicitly coarse summaries, not cloud masks.
- Nodata/failed tiles/analyst exclusions are tracked. Clouds, hangar occlusion and unknown aircraft are not automatically assessed.
- Unknown quality requires explicit override; optical TierA/model resolution gates remain. Unsupported empirical sensor/GSD gets domain warnings, not operational approval.
- Resource/time/stop checks occur between forwards; one stuck backend forward cannot be forcibly interrupted.
- Unbound external masks/sidecars fail before inference. Georeferencing failure preserves pixel boxes with null geometry/reason rather than invented coordinates.

## Fixed corrective pilot

**28training tiles** (24positive+4background,fourgroups,133labels), **15validation** (11positive+4background,twogroups,51labels), no independent evaluation access. One preregistered epoch/28updates from preservedC; SGD0.00005, frozenbackbone/FPN, nativeD4,batch1,CPU4threads,seed91,90supdate budget.

Updates33s; training/validation helper52.2s; whole workflow95.9s, peak874MiB.

| New validation only, threshold0.7 | Precision | Recall | F1 | Count MAE | AP50 |
|---|---:|---:|---:|---:|---:|
| OriginalC | 71.4% | 29.4% | 41.7% | 2.40 | 44.3% |
| Boundedpilot | 71.4% | 29.4% | 41.7% | 2.40 | 45.0% |

No useful operating-point improvement. Trainer exploratory threshold0.25 is recorded but not adopted. CandidateSHA:0c85a901562d75a0cb3490f02d44dd50b5f6e27b8575b1c1aa07fefbca5627f9. Candidate was not independently tested or promoted. OriginalC remains baseline.

The wrapper preserves inherited plus new training airfield/capture/source-hash lineage in the new checkpoint, correcting a resumed-training provenance omission without changing the original trainer/checkpoints.

Pilot backgrounds/diversity are insufficient. Next experiment: mine actual model false alarms **only from reserved training scenes**, independently review equipment/hangar negatives and crowds, keep validation-only selection, reserve a new untouched full-scene test. These exposed evaluation scenes must not become training examples.

## Fixes and verification

Regressions cover polygon clipping; tiling/coordinates/seam handling; native RGB/affine; null failure counts; partial masks/exclusions; latched backend/resource failures; preflight before legacy calls; independent workflow denominators; failed containment not scored as zero; CLI source/checkpoint/sidecar/hardlink output protection; continued-training lineage; boundedrange/ETag/content validation; redacted failures; validation-only calibration and numerical stability.

Baseline: **202 tests passed**. Final: **320 passed**, zero failures/errors/skips, including **118 new regressions**; final suite completed in approximately 10 seconds. [Final verification](aw01d/verification.json) records the full suite, import/CLI checks, preservation and diff/whitespace checks. Routine tests use synthetic fixtures/mocked HTTP. Real evaluation is separate. [Native successes and failures](aw01d/examples/README.md) use cached predictions; screenshots do not verify analytical correctness.

## Future consumer contract

| Information | Blue Elephant requirement |
|---|---|
| Geometry | Source pixel box, nullableWGS84polygon, originaltile/CRS/affine. |
| Count | Observed detections with status/coverage; null on failure; no true inventory claim. |
| Confidence | Rawscore, uncertain/rejected states, flags; calibratedprobability remains null. |
| Model | Architecture/hash, threshold, tiling and deduplication policy. |
| Imagery | URI/hash, satellite/aerialsource, acquisitionprecision/timezonebasis, native/PAN/outputspacing separately. |
| Failure | Analysisreason, unavailableareas, quality/resolution restrictions, cancellation/partialcoverage. |
| Analyst | Verification required, especially new sensors, crowded parking and incomplete scenes. |

Recommendation: **research contract available; detector not ready for unattended whole-airfield counting**. Blue Elephant and other repositories remain untouched. See [exact Windows workflows](AW01D_WORKFLOWS.md), [changed files](aw01d/changed-files.json), [attribution](aw01d/ATTRIBUTION.md) and [machine-readable metrics](aw01d/evaluation-summary.json). Downloaded data, original/new weights and runtime outputs remain ignored locally.

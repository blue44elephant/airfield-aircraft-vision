# AW-01D Windows workflows

Run these commands in **C:\Tools\Inventions\airfield-aircraft-vision**. The detector remains experimental. See [AW01D_VALIDATION.md](AW01D_VALIDATION.md) for measured results and [aw01d/](aw01d/) for frozen evidence. These workflows write only aircraft-repository research outputs; they do not integrate with Blue Elephant.

## 1. Verify the repository, environment and preserved checkpoint

The recorded starting branch is `feat/aw-01c-generalization`, HEAD `7f71434`. Inspect differences; do not switch branches or overwrite local changes.

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
git branch --show-current
git log -1 --oneline
git status --short

$awPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$awCheckpoint = 'artifacts/aw01c/resnet50-fpn/best.pt'
$awExpected = 'd2aecd3624648981f35def1fff7bb36789f9635546f4f848c719c48e647eda42'
$awActual = (Get-FileHash -LiteralPath $awCheckpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($awActual -ne $awExpected) { throw 'Preserved AW-01C checkpoint hash differs' }

$awRun = Join-Path 'artifacts/aw01d' ('manual-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $awRun | Out-Null

& $awPython -B -c "import airfield_aircraft_vision; from airfield_aircraft_vision import aw01d, robust_contracts, robust_inference, confidence_calibration, robustness_dataset, robustness_evaluation"
if ($LASTEXITCODE -ne 0) { throw 'AW-01D imports failed' }
& $awPython -B -m airfield_aircraft_vision.aw01d --help
if ($LASTEXITCODE -ne 0) { throw 'AW-01D CLI import failed' }
& .\.venv\Scripts\aircraft-vision.exe --help
if ($LASTEXITCODE -ne 0) { throw 'Existing inference CLI failed' }
~~~

Use the preserved project venv and [requirements-aw01c.txt](../requirements-aw01c.txt). Constructors load explicit local weights; these commands do not retrieve replacement checkpoints. `aw01d/preservation-before.json` records original AW-01/AW-01B/AW-01C hashes; preserve those files and all original imagery/metadata.

## 2. Run the complete offline automated suite

~~~powershell
& $awPython -B -m pytest -q -p no:cacheprovider --basetemp "$awRun/test-tmp" --junitxml "$awRun/tests.xml"
if ($LASTEXITCODE -ne 0) { throw 'Automated regression failure; inspect tests.xml' }
git diff --check
if ($LASTEXITCODE -ne 0) { throw 'Git whitespace check failed' }
~~~

Routine tests use synthetic inputs, local fixtures and mocked HTTP. They do not download imagery or require credentials. Native-grid, masks, source-coordinate reconstruction, overlapping-tile deduplication, failed/unsuitable/cancelled states and export checks establish software behavior; they do not establish real-world aircraft accuracy.

## 3. Reproduce validation-only confidence diagnostics

Use a **fresh output directory**. The recorded confidence diagnostic uses the unchanged AW-01C validation set, not new independent robustness labels.

~~~powershell
& $awPython -B -u -m airfield_aircraft_vision.aw01d calibrate --manifest data/rareplanes-aw01c/manifest-with-negatives.json --checkpoint $awCheckpoint --out "$awRun/calibration" --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Validation calibration failed' }
$awCalibration = Join-Path $awRun 'calibration/calibration.json'
~~~

The command saves raw validation predictions, temperature-grid diagnostics and leave-one-airfield-out diagnostics. A fitted temperature is descriptive; it is **not automatically applied**. The frozen raw acceptance threshold remains **0.7**, with **0.2** as the uncertain/review floor. Neither model nor thresholds are selected using robustness test labels. Calibration is conditional on proposed detections and cannot describe missed aircraft or prove an inventory complete.

## 4. Reproduce frozen robustness evaluation

The recorded runs use `artifacts/aw01d/tiles-evaluation` and `artifacts/aw01d/scene-evaluation`. Preserve those directories. Rerun into fresh paths:

~~~powershell
& $awPython -B -u -m airfield_aircraft_vision.aw01d evaluate --cohorts docs/aw01d/cohorts-tiles.json --checkpoint $awCheckpoint --calibration $awCalibration --out "$awRun/tiles-evaluation" --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Frozen tile evaluation failed' }
& $awPython -B -u -m airfield_aircraft_vision.aw01d evaluate --cohorts docs/aw01d/cohorts-scenes.json --checkpoint $awCheckpoint --calibration $awCalibration --out "$awRun/scene-evaluation" --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Frozen scene evaluation failed' }
~~~

The first cohort contains 48 fresh positive tiles from eight previously unused geographic groups. Remaining cohorts separately assess reviewed negatives, larger georeferenced scenes and independent-source NAIP windows. The evaluator fixes the AW-01C checkpoint hash and writes an evaluation lock with manifest hashes **before inference**.

Outputs include `report.json`, per-cohort summaries, raw predictions, individual scene results and GeoJSON. Metrics distinguish the unchanged baseline, safe workflow and supplementary boundary-containment heuristic. Accepted-only AP50 is unavailable; candidate-level AP50 belongs to the baseline. Failed/unsuitable/cancelled scenes retain null counts. Primary accuracy metrics use each workflow's complete-image subset; partial coverage is excluded because full-source truth may include unavailable pixels. Separate incomplete-source baseline diagnostics are coverage-confounded and must not be presented as whole-scene accuracy. Shared airfields/captures among tile crops, full scenes and empty windows prevent pooling them as independent observations.

These test labels are now exposed. Reproducing results does not create another untouched test set. Do not select models, thresholds or deduplication parameters from rerun scores.

## 5. Run one bounded native scene and export geospatial detections

Prepare source metadata from a preserved native scene manifest. This reads source files and writes a separate request description.

~~~powershell
@'
import sys
from pathlib import Path
from airfield_aircraft_vision.dataset import load_manifest, resolve_image

run = Path(sys.argv[1])
manifest_path = Path("data/rareplanes-aw01d/window-scene-manifest.json")
sample = load_manifest(manifest_path).samples[0]
(run / "source.json").write_text(sample.source.model_dump_json(indent=2), encoding="utf-8")
(run / "image-path.txt").write_text(str(resolve_image(manifest_path, sample).resolve()), encoding="utf-8")
'@ | & $awPython -B - $awRun
if ($LASTEXITCODE -ne 0) { throw 'Source description preparation failed' }

$awImage = (Get-Content -LiteralPath "$awRun/image-path.txt" -Raw).Trim()
$awStop = Join-Path (Resolve-Path $awRun) 'STOP'
Write-Host "Emergency stop file: $awStop"

& $awPython -B -u -m airfield_aircraft_vision.aw01d infer --image $awImage --source "$awRun/source.json" --checkpoint $awCheckpoint --expected-sha256 $awExpected --output "$awRun/scene.json" --geojson "$awRun/scene.geojson" --max-tiles 256 --max-pixels 25000000 --max-seconds 300 --threads 4 --allow-unknown-quality --stop-file $awStop
$awInferenceExit = $LASTEXITCODE
Get-Content -LiteralPath "$awRun/scene.json" -Raw
Write-Host "Inference process exit: $awInferenceExit"
~~~

`--allow-unknown-quality` is an explicit research override for imagery lacking a verified quality label; it does not certify clear, unobstructed imagery. Omit it for normal input gating. An external `.msk` requires its own SHA-256 via `--mask-sha256`; the evaluated native derivatives use internal masks. Original RGB pixels are neither resized nor reprojected. Boxes are restored to source-pixel coordinates; geographic polygons use the original affine/CRS. Missing georeferencing yields no invented coordinates.

Defaults are **256 tiles, 25 million source pixels, 300 seconds and four CPU threads**. Time/cancellation limits are cooperative between tile operations; initialization and an active model forward pass can exceed the nominal wall-clock limit. Native tiles are 512 pixels with 128-pixel overlap. Default deduplication is confidence-ordered IoU NMS. `--dedup boundary_containment` additionally treats some tile-edge fragments as duplicate candidates; inspect its uncertainty flags rather than assuming an exact count.

For an emergency stop, create the **printed absolute stop-file path** in another PowerShell window, or press Ctrl+C in the foreground. A cooperative-stop example from the repository is:

~~~powershell
$awLatestManual = Get-ChildItem -LiteralPath 'artifacts/aw01d' -Directory -Filter 'manual-*' | Sort-Object Name | Select-Object -Last 1
if ($null -eq $awLatestManual) { throw 'No manual run directory found' }
New-Item -ItemType File -Path (Join-Path $awLatestManual.FullName 'STOP') -ErrorAction Stop | Out-Null
~~~

Use the printed path directly if several manual runs are active. Keep the result and stop marker; use a new output directory for another run.

### Interpret results safely

Read `status`, `coverage`, `analysis_failure_reason` and `observed_count` together:

- `complete`: the accepted detection count covers all analyzable pixels under the declared input policy.
- `partial`: accepted detections cover only inspected/analyzable pixels; unavailable areas are reported.
- `failed`, `unsuitable` or `cancelled`: `observed_count` is **null**, not zero.
- `scene_total` is always null: detector outputs do not establish the true aircraft inventory.
- Accepted, uncertain and rejected candidates are distinct. Every result requires analyst verification; confidence is not a count confidence interval.

CLI exit 0 permits either complete or partial analysis; consumers must inspect JSON. Exit 2 denotes an unusable/failed/cancelled result. Invalid source JSON writes a small failed-result envelope with null counts. Never display the structural `accepted_count: 0` of an unsuccessful analysis as "zero aircraft."

## 6. Dataset provenance, cache replay and review policy

[aw01d/robustness-reservation.json](aw01d/robustness-reservation.json) fixes fresh groups/captures before predictions. Existing AW-01/AW-01B/AW-01C groups and captures are excluded. New test selection is deliberately stratified for small, partial and crowded aircraft; it is a diagnostic sample rather than a prevalence sample.

Frozen metadata, images, PNG georeferencing sidecars, complete-location COGs and derivative manifests remain under `data/rareplanes-aw01d`. Cached replay must verify checksums, reservation, CRS, affine, masks and physical annotation identities. Lossless native crops are separate derivatives. Geographic annotations are transformed to the original source pixel grid before clipping; tile/global label alignment is checked explicitly.

An annotation-free window is only a proposal. Reviewed negatives require a recorded native window, source checksum, split, reviewer and decision before model predictions. Reviews here are **Codex visual review**, not an independent expert survey. Ambiguous aircraft/shadows and clipped objects remain ignored regions. Do not silently convert them into verified empty imagery. Keep training/validation/test geographic groups separate; do not create training negatives from exposed evaluation scenes.

### Verify or replay the frozen RarePlanes dataset

The cached audit makes no network requests:

~~~powershell
& $awPython -B -m airfield_aircraft_vision.robustness_dataset audit --root data/rareplanes-aw01d --frozen docs/aw01d
if ($LASTEXITCODE -ne 0) { throw 'Frozen native dataset audit failed' }
~~~

To replay missing lossless derivatives/manifests from the existing original cache, use the same frozen recipe without enabling downloads:

~~~powershell
& $awPython -B -m airfield_aircraft_vision.robustness_dataset replay --root data/rareplanes-aw01d --frozen docs/aw01d
if ($LASTEXITCODE -ne 0) { throw 'Local frozen dataset replay failed' }
~~~

For a separate fresh reproduction directory, explicit opt-in permits only the frozen public RarePlanes URLs, under the bounded 128 MiB aggregate retrieval limit. This is a manual data-preparation step, never part of routine GUI/tests:

~~~powershell
$awReplay = 'data/rareplanes-aw01d-replay-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
& $awPython -B -m airfield_aircraft_vision.robustness_dataset replay --root $awReplay --frozen docs/aw01d --allow-download
if ($LASTEXITCODE -ne 0) { throw 'Bounded frozen reproduction failed; preserve partial outputs' }
& $awPython -B -m airfield_aircraft_vision.robustness_dataset audit --root $awReplay --frozen docs/aw01d
if ($LASTEXITCODE -ne 0) { throw 'Reproduced dataset audit failed' }
~~~

Replay verifies frozen source hashes, windows, reviews and manifests. It does not rerank scenes or interpret detector predictions. Existing differing data cause a stop; resolve the mismatch rather than replacing originals. Review records are replayed only for their exact parent bytes/native windows; new imagery requires a new review and a new evaluation reservation.

### Independent NAIP cache replay

~~~powershell
& $awPython -B -m airfield_aircraft_vision.independent_source --annotations data/rareplanes-aw01c/metadata/RarePlanes_Public_All_Annotations.geojson --output data/aw01d-independent --evidence docs/aw01d
if ($LASTEXITCODE -ne 0) { throw 'Independent source provenance or native cache verification failed' }
~~~

With the existing completed evidence/cache, this command checks local bytes, RGB/grid, masks and provenance **without network access**. A fresh cache performs the explicitly bounded anonymous retrieval. Routine automated tests never run live retrieval.

The two retrieved 1024-pixel windows use native **0.6 m NAIP aerial imagery**, not satellite imagery, from Moab (2021-09-17) and Cody (2022-07-06). They lie more than 100 km from every RarePlanes annotation. NAIP is outside the AW-01C WorldView validation sensor/resolution domain (approximately 0.311-0.393 m). This is a domain-shift check, not proof of independent satellite generalization.

The first two windows consumed **41 range requests / 10.25 MiB**. One additional, pre-reserved Cody GA-apron attempt consumed its remaining **20 requests / 5 MiB** and stopped incomplete; it produced no usable image or zero-count result. The completed and failed attempts total **61 requests / 15.25 MiB**, with headroom for an earlier header-only grid probe. Preserve [the incomplete attempt evidence](aw01d/independent-source-ga-retrieval.json). `--extra-cody-ga` refuses automatic retry after this recorded failure. Do not reset budgets to repeat that search.

Unsigned source URIs, ETags, exact windows and per-range checksums are retained. Anonymous short-lived SAS tokens stay in memory and are not written to evidence or logs. No account credentials are read. Acquisition **dates** are retained; raw STAC 16:00Z values are not presented as verified sensing times.

The [official USDA record](https://catalog.data.gov/dataset/national-agriculture-imagery-program-naip-imagery) specifies public-domain access. The [Planetary Computer collection](https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip) has a conflicting `proprietary` machine field and a "Public Domain" license link; that contradiction remains in the frozen contract. Attribute USDA NAIP, Microsoft Planetary Computer and Esri. RarePlanes imagery/annotations retain their original **CC BY-SA 4.0** attribution and derivative conditions.

No positive aircraft label was independently verified in these NAIP windows. Moab's cross-shaped object and Cody's edge object remain ambiguous/ignored. Positive-aircraft recall is therefore **undefined**, not zero or successful. Background findings apply only to reviewed, analyzable areas.

## 7. Optional bounded correction pilot

The correction pilot is separate from the frozen AW-01C baseline. Use only the reserved new **training and validation** examples; no evaluation-label tuning. Original checkpoints remain unchanged and the output directory must be fresh:

~~~powershell
& $awPython -B -u -m airfield_aircraft_vision.robustness_correction --manifest data/rareplanes-aw01d/correction-manifest.json --checkpoint $awCheckpoint --out "$awRun/correction-pilot"
if ($LASTEXITCODE -ne 0) { throw 'Bounded correction pilot failed; retain logs' }
~~~

This wrapper fixes **one epoch, learning rate 0.00005, 90 training seconds, four CPU threads and seed 91**. The time bound excludes initialization and baseline/candidate validation. It starts from the pinned AW-01C checkpoint, retains inherited training lineage and trains the existing RPN/ROI heads while the backbone/FPN remain frozen. Both validation arms use the same fixed raw **0.7** acceptance threshold.

Outputs include `pilot-lock.json`, `baseline-validation.json` (written before training), a separate `candidate.pt`, `candidate-validation.json`, `comparison.json`, `resources.json` and raw helper outputs in `training/`. Create `<pilot directory>/STOP` before training starts or `<pilot directory>/training/STOP` to stop between updates. Keep interrupted outputs; use a new output directory for a rerun.

The pilot makes no independent-test prediction and performs no automatic promotion. The frozen robustness evaluator intentionally accepts only the original AW-01C hash. A candidate requires a new untouched evaluation before a generalization claim; do not use the existing robustness test results to tune it and then call that test independent.

## 8. Future Blue Elephant consumer contract

No Blue Elephant files are changed by AW-01D. A future consumer can read the versioned `aw01d-1` JSON/GeoJSON contract:

| Information | Required interpretation |
|---|---|
| Detection geometry | WGS84 polygon when source georeferencing is valid; retain source-pixel box and tile provenance. |
| Counts/status | Show observed accepted count only with coverage/status; null on failure/unsuitability/cancellation; no inventory total claim. |
| Confidence | Keep raw score, uncertain/rejected state and flags; calibration is not automatically applied. |
| Model | Preserve architecture/version, checkpoint SHA-256, thresholds, tiling and deduplication policy. |
| Imagery | Preserve source URI/checksum, satellite or aerial sensor, acquisition precision/timezone basis, CRS/affine and native/effective resolution distinctions. |
| Unavailable areas | Display coverage, coarse unavailable tile windows, analyst exclusions and analysis failure reason. Clouds/occlusion are not automatically mapped. |
| Resolution/quality | Enforce optical Tier A and the experimental model domain; artificial upscaling does not satisfy physical-resolution requirements. |
| Analyst review | Always required, especially dense parking, tile-edge fragments, new sensors and incomplete coverage. |

Aircraft-type identification, UAV classification, SAR aircraft inference and operational inventory counting remain outside this workflow.

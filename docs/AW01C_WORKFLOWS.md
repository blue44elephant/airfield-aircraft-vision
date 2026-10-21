# AW-01C Windows reproduction

Run in **C:\Tools\Inventions\airfield-aircraft-vision**. The experiment uses native optical pixels and local CPU inference. Results and limitations are in [AW01C_VALIDATION.md](AW01C_VALIDATION.md). These commands do not modify the other Blue Elephant projects.

## 1. Verify the repository and environment

The recorded starting point is branch `feat/aw-01c-generalization`, commit `137cdf6`. Preserve any current working changes. No branch switch or Git write is part of this workflow.

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
git branch --show-current
git log -1 --oneline
git status --short
$awPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $awPython -B -m airfield_aircraft_vision.aw01c --help
~~~

The existing project venv retains the AW-01B package versions. [requirements-aw01c.txt](../requirements-aw01c.txt) includes those pins and `psutil==7.2.2` for CPU/memory measurements. Official NumPy 2.4.2 is installed inside the project venv; changing global Python is unnecessary.

For a **new environment**, install the official CPU Torch pair before the remaining pinned dependencies:

~~~powershell
py -m venv .venv
$awPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $awPython -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
if ($LASTEXITCODE -ne 0) { throw 'CPU Torch installation failed' }
& $awPython -m pip install -r requirements-aw01c.txt
if ($LASTEXITCODE -ne 0) { throw 'Pinned dependency installation failed' }
& $awPython -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Local package installation failed' }
~~~

Model constructors never retrieve upstream weights implicitly. Dataset preparation explicitly downloads only bounded public RarePlanes metadata/tiles when their verified cache is unavailable.

## 2. Preserve the lineage and reserve the independent cohort

Required original local artifacts:

- `data/rareplanes/manifest.json`: original AW-01 manifest.
- `data/rareplanes-aw01b/manifest-balanced.json` plus its `metadata/` and cached `images/`.
- `artifacts/aw01b/balanced_frozen_sgd/best.pt`: preserved AW-01B comparison checkpoint.
- `data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth`: original official COCO initializer.

The new reservation fixes **68 tiles from nine previously unexposed geographic groups**, using official-test-only membership and seeded image IDs before model experiments. Their images/labels must not guide preparation, negatives, thresholds or architecture selection. The exposed AW-01B test is regression data.

Create a separate data root and install the exact tracked reservation **before preparation or training**. Existing differing files cause an explicit stop.

~~~powershell
$awData = 'data/rareplanes-aw01c'
New-Item -ItemType Directory -Force $awData | Out-Null
$awReservation = Join-Path $awData 'test-reservation.json'
$awReservationSource = 'docs/aw01c/test-reservation.json'
if (Test-Path -LiteralPath $awReservation) {
    $awExistingHash = (Get-FileHash -LiteralPath $awReservation -Algorithm SHA256).Hash
    $awRecordedHash = (Get-FileHash -LiteralPath $awReservationSource -Algorithm SHA256).Hash
    if ($awExistingHash -ne $awRecordedHash) {
        throw 'Existing reservation differs; do not overwrite it'
    }
} else {
    Copy-Item -LiteralPath $awReservationSource -Destination $awReservation
}
& $awPython -B -m airfield_aircraft_vision.aw01c prepare --baseline data/rareplanes-aw01b/manifest-balanced.json --legacy data/rareplanes/manifest.json --root $awData
if ($LASTEXITCODE -ne 0) { throw 'Bounded preparation failed' }
~~~

Current CLI preparation defaults are seed 41, 32 training groups, up to 16 tiles per training group, four workers and a **192 MiB image-download budget**. A fresh root separately retrieves the verified rich training COCO annotations (33,539,381 bytes) under a **40 MiB metadata cap**, validates the full SHA-256 and joins all image/annotation IDs and boxes before preparing samples. This rich metadata is included in fresh-root lineage even when the AW-01B metadata cache lacks it; the recorded existing frozen manifest is not rewritten. Original training/validation membership is preserved; additional training choices alternate small-box, crowded and seeded ordering, with capture round robin. No test annotation is used to rank training data.

`request.json` fixes the preparation configuration. `protocol.json` fixes assignments/selected IDs. Original manifest/reservation/protocol/metadata hashes, geographic group exclusions, source checksums and cross-split footprint checks protect the lineage. Exact cached preparation verifies its contracts; another configuration requires a new isolated root. The budget covers new downloaded bytes, not all copied cached data, checkpoints or temporary files.

Minute timestamps retain their original strings and **assumed UTC** basis. Source panchromatic resolution, delivered pixel spacing, georeferencing, sidecars, licensing and acquisition provenance remain distinct. No pixel resize, reprojection or artificial resolution increase occurs.

## 3. Attach explicitly reviewed negative crops

RarePlanes' supplied tiles are positive samples. Absence of a box is insufficient to declare an empty apron.

AW-01C creates native crops of at most 256 pixels from **training/validation parents only**, excluding all positive/ignored labels plus a 32-pixel halo. Each accepted crop has a documented visual inspection bound to its exact parent checksum and native window. The recorded reviews identify **Codex visual review**; they are not a claim of a separate human survey. No script automatically approves a crop.

For reproduction, replay the preserved reviews only against the same verified source bytes:

~~~powershell
& $awPython -B -m airfield_aircraft_vision.aw01c attach-negatives --manifest data/rareplanes-aw01c/manifest.json --reviews docs/aw01c/negative_reviews.json --output data/rareplanes-aw01c/manifest-with-negatives.json
if ($LASTEXITCODE -ne 0) { throw 'Negative review attachment failed' }
$awManifest = 'data/rareplanes-aw01c/manifest-with-negatives.json'
~~~

For additional reviews, inspect every proposed crop at native display scale before recording a decision. Review metadata must include the parent ID/checksum, exact window, category, reviewer, aware timestamp and notes. Categories distinguish empty apron from difficult/other background; roads, trees and buildings are not relabeled as empty aprons. Uncertain crops must be excluded. Fully valid raster pixels are required; no-data crops are rejected.

Attachment writes separate lossless GeoTIFF derivatives with adjusted native affine transforms and parent/review provenance. Original PNGs/sidecars and the base manifest remain unchanged. Exact replay checks existing derivative bytes; differing approvals or files are rejected.

These are small reviewed background crops within positive scenes. They do **not** establish independently surveyed full empty airfields, and the new independent cohort contains positive tiles only. Negative false-alarm findings come from the separate validation backgrounds.

## 4. Obtain only the explicit ResNet initializer if needed

The MobileNet initializer remains unchanged. The alternative uses one official TorchVision COCO ResNet-50 FPN artifact:

- [Official model specification](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).
- Explicit local filename: `data/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`.
- Loader validates the official SHA-256 prefix `258fb6c6`.
- Code uses the official FrozenBatchNorm-compatible architecture and copies COCO airplane/background head rows. No random initialization is substituted for the pretrained detector.

If that file is absent, this **single explicit bounded download** is sufficient. No other model downloads are needed:

~~~powershell
@'
from pathlib import Path
from airfield_aircraft_vision.architecture_backend import RESNET_WEIGHTS_URL
from airfield_aircraft_vision.dataset import BoundedDownload
from airfield_aircraft_vision.preprocessing import sha256_file

path = Path("data/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
path.parent.mkdir(parents=True, exist_ok=True)
BoundedDownload(192 * 1024**2).fetch(
    RESNET_WEIGHTS_URL, path, max_file=192 * 1024**2
)
digest = sha256_file(path)
if not digest.startswith("258fb6c6"):
    raise ValueError("Official ResNet initializer checksum prefix differs")
print(digest)
'@ | & $awPython -B -
if ($LASTEXITCODE -ne 0) { throw 'Explicit pretrained artifact acquisition failed' }
~~~

The actual measured feasibility, resource usage and source/checkpoint hashes are retained with validation evidence. TorchVision code licensing and pretrained-data conditions must be considered separately from RarePlanes' CC BY-SA 4.0 image/annotation terms; no automatic model redistribution permission is asserted.

## 5. Run the two controlled training arms

Use a new output root. Run the CPU jobs **sequentially**.

~~~powershell
$awRun = 'artifacts/aw01c-rerun-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Force $awRun | Out-Null
& $awPython -B -u -m airfield_aircraft_vision.aw01c train --manifest $awManifest --weights data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth --architecture fasterrcnn_mobilenet_v3_large_fpn --out "$awRun/expanded-mobilenet" --epochs 3 --lr 0.0025 --threads 4 --max-seconds 900
if ($LASTEXITCODE -ne 0) { throw 'Expanded MobileNet training failed' }
& $awPython -B -u -m airfield_aircraft_vision.aw01c train --manifest $awManifest --weights data/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth --architecture fasterrcnn_resnet50_fpn --out "$awRun/resnet50-fpn" --epochs 3 --lr 0.0025 --threads 4 --max-seconds 1500
if ($LASTEXITCODE -ne 0) { throw 'ResNet training failed' }
~~~

Both arms use the same manifest, native D4 flips/90-degree rotations, seed 41, airfield round-robin sampling, batch one and SGD with momentum 0.9/weight decay 0.0005. The entire convolutional backbone **and FPN** remain frozen; RPN/ROI heads train. Learning rate multiplies by 0.3 after each epoch. Gradient norm is limited to 10. The shared training limits are 1,000 pre-NMS proposals, 512 post-NMS proposals and 128 ROI samples; evaluation retains its configured inference proposal defaults.

Training accepts at most eight epochs/eight CPU threads and requires native optical inputs no larger than 512 pixels. Native smaller background crops are padded by the existing raster reader, never stretched into higher physical resolution. Inputs are uint8 RGB divided by 255 and normalized once. Ignored training regions are rejected rather than silently treated as background.

`run.json` records the PID/configuration/input hashes. `progress.json`, `history.json`, `best-validation.json`, `best.pt` and `resources.json` record measured progress, validation selection, elapsed time, CPU time and peak process memory. Memory is measured, not a guaranteed hard allocation cap.

The time limit is checked between training updates. Initialization, the current update, validation and checkpoint writing may make total command time exceed the configured loop limit. A stopped partial epoch is identified in history/resources; it is not represented as a completed epoch.

To request a cooperative stop of one known run:

~~~powershell
New-Item -ItemType File -Path "$awRun/resnet50-fpn/STOP" | Out-Null
~~~

The current update can finish, followed by validation/checkpoint bookkeeping. Ctrl+C stops a foreground command; retain any completed checkpoint/logs and use a new output directory for another run.

Each epoch evaluates **validation only**. Threshold grid and selection are inherited from AW-01B: maximum validation F1, then lower count MAE, then higher threshold. No independent-test labels or predictions select architecture, epoch, learning rate or confidence.

## 6. Freeze selection, then evaluate the independent cohort once

Run this only after all model/threshold decisions are final. Do not reuse independent results to tune another candidate and then call the same cohort independent.

~~~powershell
& $awPython -B -u -m airfield_aircraft_vision.aw01c compare --manifest $awManifest --baseline-checkpoint artifacts/aw01b/balanced_frozen_sgd/best.pt --candidates "$awRun/expanded-mobilenet" "$awRun/resnet50-fpn" --out "$awRun/comparison" --regression-manifest data/rareplanes-aw01b/manifest-balanced.json --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Locked comparison failed; retain its outputs and investigate' }
~~~

A fresh comparison directory is mandatory, including after a partially failed run. Before test parsing/prediction, the workflow verifies candidate provenance and hashes, retains AW-01B's stored validation threshold, chooses the candidate from validation metrics and writes `selection-lock.json`. Each model is inferred once per cohort. The zero-count baseline requires no model inference.

Outputs include:

- Independent and separately labeled AW-01B regression P/R/F1/AP50, count MAE/bias, FP/FN and zero-baseline comparison.
- Source-pixel size, annotated crowding, airfield, geography, weather and explicitly reviewed negative strata.
- Raw scored predictions and fixed-operating-point per-image match records.
- Native-size overlays for the first 12 independent image IDs per model, using the **same cached predictions** as metrics. No second model pass generates examples.
- Source provenance, selection/checkpoint hashes, sampling limitations and dataset attribution.

AP50 is the existing 101-point IoU-0.5 measure down to score floor 0.005, not multi-IoU COCO AP. Confidence is uncalibrated. Bounding-box size is a visibility-dependent source-pixel proxy, not a verified aircraft class or physical wingspan. Source-correlated per-tile counts cannot be summed into an airport inventory. Small strata and the limited geography require cautious descriptive interpretation.

Cyan marks labels, green matched detections, red unmatched detections and gray ignored regions. Visual overlays do not independently verify analytical correctness. Original source files are unchanged.

## 7. Offline tests and final review

Routine tests use synthetic rasters, mock predictions and mocked HTTP; they do not download images or models.

~~~powershell
New-Item -ItemType Directory -Force artifacts/aw01c | Out-Null
& $awPython -B -m pytest -q -p no:cacheprovider --basetemp=artifacts/aw01c/test-tmp --junitxml=artifacts/aw01c/tests.xml
if ($LASTEXITCODE -ne 0) { throw 'Automated tests failed' }
git diff --check
git diff --stat
git status --short
~~~

All generated data/checkpoints/logs stay in ignored `data/` and `artifacts/`. Selected evidence/manifests/reports under `docs/aw01c/` remain reviewable. Retain RarePlanes attribution and share-alike terms for derived imagery. Nothing is committed, pushed, merged or integrated into another application automatically.

## 8. Launch inference with the preserved selected model

This uses a validation image, so it does not repeat the independent evaluation. The installed `aircraft-vision` command and `python -m airfield_aircraft_vision` both support the existing inference workflow.

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
$awPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
@'
from pathlib import Path
import json
from airfield_aircraft_vision.dataset import load_manifest, resolve_image

manifest_path = Path("data/rareplanes-aw01c/manifest-with-negatives.json")
manifest = load_manifest(manifest_path)
sample = next(s for s in manifest.samples if s.split == "validation" and s.boxes)
folder = Path("artifacts/aw01c")
folder.mkdir(parents=True, exist_ok=True)
(folder / "manual-source.json").write_text(sample.source.model_dump_json(indent=2), encoding="utf-8")
(folder / "manual-request.json").write_text(json.dumps({"image": str(resolve_image(manifest_path, sample))}), encoding="utf-8")
'@ | & $awPython -B -
if ($LASTEXITCODE -ne 0) { throw 'Source preparation failed' }
$awImage = (Get-Content -Raw -LiteralPath artifacts/aw01c/manual-request.json | ConvertFrom-Json).image
$awOutput = 'artifacts/aw01c/manual-inference-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
& $awPython -B -m airfield_aircraft_vision infer --image $awImage --source artifacts/aw01c/manual-source.json --checkpoint artifacts/aw01c/resnet50-fpn/best.pt --output $awOutput --allow-unknown-quality --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Inference failed' }
Get-Content -Raw -LiteralPath $awOutput
~~~

For a freshly trained reproduction, substitute `"$awRun/resnet50-fpn/best.pt"` as the checkpoint. The original measured checkpoint remains in `artifacts/aw01c/resnet50-fpn/best.pt`.

Omitting `--confidence` uses the checkpoint's validation-selected threshold (0.7 for the recorded model). The source's unknown quality is explicitly acknowledged by `--allow-unknown-quality`; it is not relabeled acceptable. Existing capability restrictions still block unsuitable inputs. Output includes boxes, scores, counts and source/model/geospatial provenance.

The recorded CLI smoke returned one aircraft from a validation image with `resize=false`, `reprojection=false`, threshold 0.7 and the frozen checkpoint SHA. See [verification.json](aw01c/verification.json).

Timed training is not promised to produce bit-identical weights on other hardware: the recorded ResNet run performed two complete 467-update epochs plus 170 updates of epoch three. Replay the preserved checkpoint and raw predictions to reproduce the reported comparison without selecting another model from exposed test results.

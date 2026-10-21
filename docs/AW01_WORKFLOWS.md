# AW-01 reproducible workflows (Windows PowerShell)

Run in the aircraft repository. No other project is required.

## Install

Prefer Python 3.12 in an isolated environment (the recorded run used existing Python 3.14 CPU packages).

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,vision]'
python -m airfield_aircraft_vision --help
New-Item -ItemType Directory -Force artifacts | Out-Null
python -m pytest -q -p no:cacheprovider --basetemp=artifacts/test-tmp
~~~

GPU users must install the official compatible PyTorch build and explicitly use `--device cuda`.
No automatic dependency, provider or credential installation occurs during inference.

## Prepare a bounded real sample

~~~powershell
python -m airfield_aircraft_vision prepare --root data/rareplanes --seed 17 --train-groups 3 --validation-groups 2 --test-groups 3 --tiles-per-group 3
python -m airfield_aircraft_vision fetch-weights --output data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth
~~~

This requests at most 24 selected tiles plus sidecars, license and bounded annotation/CSV metadata.
Each preparation run has a 128 MiB network budget. The explicit initial-weight request is about 75 MiB and has a
separate budget. No credentials are read. Each request has a timeout and at most one retry.
Ctrl+C stops execution. Completed originals are retained and reused; incomplete transfers are never installed.
Use another root for a different split, or inspect/delete only your own experiment outputs deliberately.

## Train and evaluate

~~~powershell
python -m airfield_aircraft_vision train --manifest data/rareplanes/manifest.json --init-weights data/weights/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth --output artifacts/aw01-3epoch.pt --epochs 3 --seed 17 --threads 4 --device cpu
python -m airfield_aircraft_vision evaluate --manifest data/rareplanes/manifest.json --checkpoint artifacts/aw01-3epoch.pt --split validation --outdir artifacts/aw01-validation --confidence 0.5 --threads 4
python -m airfield_aircraft_vision evaluate --manifest data/rareplanes/manifest.json --checkpoint artifacts/aw01-3epoch.pt --split test --outdir artifacts/aw01-test --confidence 0.5 --threads 4
~~~

Output paths must be fresh. The training backbone and its normalization statistics are frozen. The single-class head
starts from the COCO airplane class; ROI and proposal layers are fine-tuned. Fixed final-epoch selection and
confidence threshold are recorded. Neither validation nor test images enter training; the CLI checks checkpoint
airfields, captures and image checksums before evaluation. Do not tune on test outputs.

Metrics: confidence-ordered one-to-one matching at IoU 0.5; micro precision/recall; per-tile count MAE, bias, RMSE
and absolute error. Unmatched/ignored predictions and missed ground-truth indices are saved. These are not COCO mAP
or whole-airfield inventory metrics. Missing denominators produce null, not fabricated perfect scores.

Each evaluation directory contains:
- `metrics.json`: parameters, source/checkpoint hashes, per-image matching and aggregate metrics;
- `<image>.json`: detections, rejected candidates, source and model provenance;
- `<image>.png`: separate native-size overlay (cyan truth/FN, green matches, red unmatched detections);
- `ATTRIBUTION.txt`: source credit, changes, license and legend.

## Infer on local imagery

Supply a `RasterSource` JSON (see `optical_contracts.py`). It must declare source SHA-256, actual dimensions,
sensor domain, trustworthy physical GSD, quality, and optional original CRS/affine/acquisition precision.
For a prepared sample, extract its exact metadata:

~~~powershell
$manifest = Get-Content data/rareplanes/manifest.json -Raw | ConvertFrom-Json
$sample = $manifest.samples | Where-Object split -EQ test | Select-Object -First 1
$sample.source | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8NoBOM artifacts/source.json
python -m airfield_aircraft_vision infer --image "data/rareplanes/$($sample.path)" --source artifacts/source.json --checkpoint artifacts/aw01-3epoch.pt --output artifacts/local-inference.json --tile-size 512 --overlap 128 --confidence 0.5 --allow-unknown-quality
~~~

The JSON extraction uses PowerShell 7 `utf8NoBOM`; Windows PowerShell 5 users can instead write with
`[IO.File]::WriteAllText((Join-Path (Get-Location) 'artifacts/source.json'), ($sample.source | ConvertTo-Json -Depth 20))`.

Only explicit uint8 three-band RGB is accepted. No automatic band stretching or 16-bit conversion.
Tiles preserve native pixels, with padding at edges. TorchVision resizing is disabled in both train/eval modes.
The model still has internal feature-map downsampling; this does not alter the source or imply added resolution.
Bounds of 0.2?1 m are experimental policy limits, not empirically validated resolution coverage.

No scene with entirely invalid pixels produces a misleading zero count. Partial nodata and unknown quality carry warnings.
Unknown/date-only sensing time cannot be exported as an exact-time VISION-0 result. Where a provider-confirmed UTC basis, time precision and georeferencing
support it, a compatible reconciled `InferenceResult` is included alongside the richer experimental envelope.

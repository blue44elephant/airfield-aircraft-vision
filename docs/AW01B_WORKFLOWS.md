# AW-01B Windows reproduction

Run only in `C:\Tools\Inventions\airfield-aircraft-vision`. No other project is used.

## Environment

The final experiment uses the project-local venv, CPU Torch 2.10.0 / TorchVision 0.25.0 and official NumPy 2.4.2.
The inherited system NumPy 1.26.4 experimental build is superseded only inside this venv; global packages remain unchanged.
[Official NumPy 2.4.2 release](https://pypi.org/project/numpy/2.4.2/).

For an existing AW-01 environment:

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
.\.venv\Scripts\python.exe -m pip --isolated install --no-deps --only-binary=:all: --index-url https://pypi.org/simple numpy==2.4.2
$awPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $awPython -m airfield_aircraft_vision.aw01b --help
~~~

For a fresh environment, create a venv, install the official CPU Torch/TorchVision pair, then install
`requirements-aw01b.txt` and `pip install --no-deps -e .`. Use Python 3.11?3.14 supported by the pinned dependencies.

## Diagnose and prepare

Choose fresh output directories. Source data and the AW-01 checkpoint are retained.

~~~powershell
$awRun = 'artifacts/aw01b-rerun-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
New-Item -ItemType Directory -Force $awRun | Out-Null
& $awPython -m airfield_aircraft_vision.aw01b diagnose --outdir "$awRun/diagnosis" --steps 80 --threads 4
& $awPython -m airfield_aircraft_vision.aw01b prepare --root data/rareplanes-aw01b
& $awPython -m airfield_aircraft_vision.aw01b broaden-validation --manifest data/rareplanes-aw01b/manifest.json
~~~

Preparation is bounded by 16 training, four initial validation and four fresh test geographic groups,
with up to 12/8/8 native tiles per group. Some groups have fewer tiles.
A separate revision adds two unused validation groups for larger objects and cloud/haze metadata,
at up to eight tiles each. Original training/test samples remain identical.

First-stage image downloads have a 128 MiB budget and four workers; validation additions have a separate
16 MiB budget. Cached metadata is reused. No original full scene, credentials or recursive bucket is used.
The acquisition strings/precision, supplied georeferencing, source resolutions, licenses and checksums remain available.
Completed original files are reused. Ctrl+C stops a foreground command; active HTTP requests have bounded timeouts.
Use another root for a different experiment rather than replacing frozen manifests.

## Train the two predeclared candidates

~~~powershell
& $awPython -m airfield_aircraft_vision.aw01b train --profile frozen_sgd --outdir "$awRun/frozen_sgd" --epochs 6 --threads 4
if ($LASTEXITCODE -ne 0) { throw 'Frozen training failed' }
& $awPython -m airfield_aircraft_vision.aw01b train --profile fpn_adamw --outdir "$awRun/fpn_adamw" --epochs 6 --threads 4
if ($LASTEXITCODE -ne 0) { throw 'FPN training failed' }
~~~

Both start from the same checked COCO initializer and native-pixel model.
`frozen_sgd`: frozen backbone, SGD 0.005, momentum 0.9, weight decay 0.0005.
`fpn_adamw`: frozen convolutional body, trainable feature pyramid, AdamW 0.0003, weight decay 0.0001.
Both: batch two, seed 23, native flips/90? rotations, no resizing, fixed normalization, frozen backbone normalization statistics,
gradient-norm limit 10 and six epochs.

Each epoch is evaluated on validation only. The threshold grid is
0.05/0.10/0.15/0.20/0.25/0.30/0.40/0.50/0.60/0.70/0.80.
Select highest validation F1, then lower count MAE, then higher threshold. AP50 is reported, not the selection target.
`best.pt`, `best-validation.json` and `history.json` are retained for each candidate.

## Lock selection and evaluate independent imagery

~~~powershell
& $awPython -m airfield_aircraft_vision.aw01b compare --candidate-dirs "$awRun/frozen_sgd" "$awRun/fpn_adamw" --outdir "$awRun/comparison" --threads 4
~~~

The comparison writes a selection lock before test evaluation.
It evaluates the selected model, AW-01 at 0.5, AW-01 at its validation-selected threshold and a zero-detection baseline.
Fresh test is the confirmatory cohort; the previously exposed AW-01 test is a separate regression cohort.
No test-set threshold search or model selection occurs.

Outputs include precision, recall, F1, AP50, count errors, per-group metrics, size/weather strata,
raw scored predictions, per-image result JSON and every native-size test overlay.
Cyan=ground truth/FN; green=matched detection; red=unmatched detection.
Counts are per tile and must not be summed as an airfield inventory; tiles may overlap within a split.

## Inference with the selected model

Use the existing `infer` workflow with a source JSON and the chosen `best.pt`.
If `--confidence` is omitted, inference uses the checkpoint's validation-selected threshold;
old AW-01 checkpoints retain the 0.5 fallback. Explicit `--confidence` overrides the default.
No artificial upscaling or aircraft-type classification is added.

## Tests and review

~~~powershell
New-Item -ItemType Directory -Force artifacts/aw01b | Out-Null
& $awPython -m pytest -q -p no:cacheprovider --basetemp=artifacts/aw01b/test-tmp --junitxml=artifacts/aw01b/tests.xml
git diff --check
~~~

Routine tests use synthetic data and mocked downloads. No satellite search or model/data download occurs.
The original checkpoint and AW-01 reports remain unchanged. Nothing is automatically committed, pushed or merged.

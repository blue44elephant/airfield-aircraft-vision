# Airfield Aircraft Vision

Standalone, resolution-aware aircraft vision prototype. Integration with Blue Elephant, Epic Archer or
`sentinel-ship-motion` is through versioned data contracts; those repositories are not modified.

## Current scope: AW-01 / AW-01B

- Reuses VISION-0's conservative optical/SAR capability router and validated count contracts.
- Experimental high-resolution optical aircraft boxes, scores and total visible-detection counts.
- Native-pixel overlapping tiles, scene-wide NMS and source-affine geographic export.
- Bounded RarePlanes preparation with separate airfield/capture/checksum groups.
- Reproducible COCO-initialized TorchVision training, unseen-airfield evaluation and native-size overlays.
- Explicit source/model checksums, time precision, RGB spacing, physical resolution and quality limitations.
- Offline regression tests; no automatic network requests during inference or tests.

Only generic aircraft detection is implemented. Aircraft type/heading, UAV model classification, SAR aircraft
detection and lower-resolution aircraft inference remain unavailable. Results are experimental, not aircraft inventory.

## Quick start (PowerShell)

~~~powershell
Set-Location 'C:\Tools\Inventions\airfield-aircraft-vision'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,vision]'
python -m airfield_aircraft_vision --help
New-Item -ItemType Directory -Force artifacts | Out-Null
python -m pytest -q -p no:cacheprovider --basetemp=artifacts/test-tmp
~~~

See [exact preparation/training/inference commands](docs/AW01_WORKFLOWS.md),
[audit](docs/AW01_AUDIT.md), [RarePlanes terms and limits](docs/AW01_DATASET.md),
[measured validation](docs/AW01_VALIDATION.md) and
[Blue Elephant integration recommendations](docs/AW01_BLUE_ELEPHANT_INTEGRATION.md).
Downloaded data, weights and runtime artifacts are ignored by Git.

## Scientific rule

Capabilities follow the defensible physical information content of the source scene, after quality penalties.
Artificial enlargement and super-resolution never improve a capability tier. Unknown or unreliable physical
resolution blocks automatic routing. Native source pixels are never replaced with a sharpened/upscaled substitute.

Original [architecture](docs/ARCHITECTURE.md), [dataset specification](docs/DATASET_SPEC.md) and
[capability gating](docs/CAPABILITY_GATING.md) remain the design foundation. AW-01 is a bounded implementation of
their high-resolution optical path; it does not claim to implement the full roadmap.

## AW-01B validation

The detector was retrained on 174 native tiles from 16 airfield groups with validation-only checkpoint/threshold selection.
Fresh-test recall is 21.0%, precision 48.6% and count MAE 1.93 per tile; generalization remains inadequate for operational use.
See [diagnosis and measured comparisons](docs/AW01B_VALIDATION.md), [Windows reproduction](docs/AW01B_WORKFLOWS.md)
and [reviewed successes/failures](docs/examples/aw01b/README.md). The original AW-01 checkpoint and reports remain intact.

# AW-01 repository audit

Audit started 2026-10-21 in `C:\Tools\Inventions\airfield-aircraft-vision`.
Origin verified as `github.com/blue44elephant/airfield-aircraft-vision.git`.
Branch `main`, HEAD `a857820` (Complete initial aircraft vision architecture), initially clean.
No branch switch, commit, push or merge. No changes to Blue Elephant, Epic Archer or Sentinel Ship Motion.

## Reusable VISION-0 components

- `capability.py`: functioning conservative physical-resolution and sensor router. Reused unchanged.
- `schemas.py`: functioning Pydantic detection, uncertainty and count reconciliation contracts. Reused unchanged.
- Provider interface, annotation schema, source-group split specification and integration boundary: retained.
- High-resolution inference, preprocessing, postprocessing and geospatial-export modules were docstring-only boundaries.
- Baseline: 30 tests passed. No detector, weights, image downloader, training loop or metric evaluator existed.
- Repository has no GUI. AW-01 adds a standalone CLI/library, not a new desktop application.

## Constraints reported before implementation

CPU-only Torch 2.10.0 / TorchVision 0.25.0; no cached model weights or local dataset.
The supplied filesystem sandbox allowed writes only to Blue Elephant; explicit target-repository write/network approval was requested.
Anonymous network sockets initially failed in the sandbox and succeeded after approved escalation.

RarePlanes access and its bucket LICENSE were verified before the experiment. Only a small seeded selection is prepared;
no recursive bucket or full-scene download. Published RarePlanes experimental checkpoints use a different framework/configuration
and are about 330 MB each; their availability is not proof of compatibility with this implementation.

The system Python is 3.14.3 with a nonstandard NumPy 1.26.4 Windows build that emits numerical/runtime warnings.
Tests and actual execution results are recorded separately; this environment is unsuitable as a production reference.
Use an isolated supported Python environment for subsequent experiments.

See [dataset assessment](AW01_DATASET.md), [commands](AW01_WORKFLOWS.md) and [measured validation](AW01_VALIDATION.md).

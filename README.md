# Airfield Aircraft Vision

Standalone, resolution-aware computer-vision foundation for aircraft and airfield activity analysis. The project is deliberately independent of Epic Archer (EA) and `sentinel-ship-motion`; integration is through versioned data contracts only.

## Current scope: VISION-0

Implemented:

- a conservative optical/SAR capability router;
- provider-neutral scene metadata and search interfaces;
- Pydantic inference output contracts with count reconciliation;
- a machine-readable aircraft annotation schema;
- architecture, dataset, capability, integration, and roadmap documents;
- unit tests for routing boundaries and output invariants.

Not implemented: provider credentials, imagery downloads, model inference, training, synthetic degradation, performance claims, operational EA integration, or a SAR detector.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## Scientific rule

Capabilities follow the best defensible physical information content of the source scene, after quality penalties. Resizing, artificial upscaling, and super-resolution never improve a capability tier. Unknown or unreliable resolution metadata blocks automatic model routing.

See [Architecture](docs/ARCHITECTURE.md), [Capability gating](docs/CAPABILITY_GATING.md), and [Roadmap](docs/ROADMAP.md).


# Workflows

The AW-01 executable workflows live in the installable package so imports, CLI and tests share implementation.

~~~powershell
python -m airfield_aircraft_vision --help
~~~

Commands: `prepare`, `fetch-weights`, `train`, `evaluate`, `infer`.
See [AW01_WORKFLOWS.md](../docs/AW01_WORKFLOWS.md) for complete Windows commands and bounds.
No imagery/provider access occurs implicitly during routine tests or inference.

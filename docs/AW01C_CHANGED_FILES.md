# AW-01C changed-file inventory

Baseline: 137cdf6, branch feat/aw-01c-generalization. No commit, push, merge or branch switch.

3 previously tracked files modified; 41 new reviewable files. All paths are relative to the aircraft repository.

## Modified files

- [README.md](../README.md)
- [pyproject.toml](../pyproject.toml)
- [src/airfield_aircraft_vision/torch_backend.py](../src/airfield_aircraft_vision/torch_backend.py)

The existing runtime change is the compatible architecture dispatcher in torch_backend.py. README and optional experiment dependencies are updated. Existing capability, schemas, image preprocessing, inference, original dataset/training/evaluation and AW-01B source files remain unchanged.

## Added source

- [src/airfield_aircraft_vision/architecture_backend.py](../src/airfield_aircraft_vision/architecture_backend.py)
- [src/airfield_aircraft_vision/aw01c.py](../src/airfield_aircraft_vision/aw01c.py)
- [src/airfield_aircraft_vision/generalization_comparison.py](../src/airfield_aircraft_vision/generalization_comparison.py)
- [src/airfield_aircraft_vision/generalization_dataset.py](../src/airfield_aircraft_vision/generalization_dataset.py)
- [src/airfield_aircraft_vision/generalization_metrics.py](../src/airfield_aircraft_vision/generalization_metrics.py)
- [src/airfield_aircraft_vision/generalization_training.py](../src/airfield_aircraft_vision/generalization_training.py)
- [src/airfield_aircraft_vision/negative_crops.py](../src/airfield_aircraft_vision/negative_crops.py)

## Added tests

- [tests/test_architecture_backend.py](../tests/test_architecture_backend.py)
- [tests/test_generalization_comparison.py](../tests/test_generalization_comparison.py)
- [tests/test_generalization_dataset.py](../tests/test_generalization_dataset.py)
- [tests/test_generalization_metrics.py](../tests/test_generalization_metrics.py)
- [tests/test_generalization_training.py](../tests/test_generalization_training.py)
- [tests/test_negative_crops.py](../tests/test_negative_crops.py)

## Added documentation and examples

- [docs/AW01C_CHANGED_FILES.md](AW01C_CHANGED_FILES.md)
- [docs/AW01C_VALIDATION.md](AW01C_VALIDATION.md)
- [docs/AW01C_WORKFLOWS.md](AW01C_WORKFLOWS.md)
- [docs/aw01c/ATTRIBUTION.md](aw01c/ATTRIBUTION.md)
- [docs/aw01c/annotation_review.json](aw01c/annotation_review.json)
- [docs/aw01c/architecture-feasibility.json](aw01c/architecture-feasibility.json)
- [docs/aw01c/aw01b_regression-raw-predictions.json](aw01c/aw01b_regression-raw-predictions.json)
- [docs/aw01c/comparison.json](aw01c/comparison.json)
- [docs/aw01c/dataset_manifest.json](aw01c/dataset_manifest.json)
- [docs/aw01c/environment.json](aw01c/environment.json)
- [docs/aw01c/failure_review.json](aw01c/failure_review.json)
- [docs/aw01c/independent_test-raw-predictions.json](aw01c/independent_test-raw-predictions.json)
- [docs/aw01c/negative_reviews.json](aw01c/negative_reviews.json)
- [docs/aw01c/pixel_audit.json](aw01c/pixel_audit.json)
- [docs/aw01c/preservation-after.json](aw01c/preservation-after.json)
- [docs/aw01c/preservation-before.json](aw01c/preservation-before.json)
- [docs/aw01c/pretrained-feasibility.json](aw01c/pretrained-feasibility.json)
- [docs/aw01c/protocol.json](aw01c/protocol.json)
- [docs/aw01c/selection-lock.json](aw01c/selection-lock.json)
- [docs/aw01c/test-reservation.json](aw01c/test-reservation.json)
- [docs/aw01c/training_histories.json](aw01c/training_histories.json)
- [docs/aw01c/verification.json](aw01c/verification.json)
- [docs/examples/aw01c/README.md](examples/aw01c/README.md)
- [docs/examples/aw01c/false-positives.png](examples/aw01c/false-positives.png)
- [docs/examples/aw01c/index.json](examples/aw01c/index.json)
- [docs/examples/aw01c/missed-aircraft.png](examples/aw01c/missed-aircraft.png)
- [docs/examples/aw01c/success.png](examples/aw01c/success.png)

## Added environment pins

- [requirements-aw01c.txt](../requirements-aw01c.txt)

## Ignored local outputs

- data/rareplanes-aw01c/: frozen reservation/protocol/manifests, bounded public metadata/images/sidecars, visually reviewed native GeoTIFF derivatives.
- data/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth: the single additional official initializer.
- artifacts/aw01c/expanded-mobilenet/ and artifacts/aw01c/resnet50-fpn/: selected checkpoints, progress, histories and resource records.
- artifacts/aw01c/comparison/: frozen selection, once-collected raw predictions, complete metrics and native examples.
- artifacts/aw01c/experiment-code/: exact executed trainer snapshot.
- artifacts/aw01c/tests.xml, test temporary directories, negative-review sheets/rejected-crop evidence, inference startup smoke outputs.

Original AW-01/AW-01B checkpoints/manifests/experiments are preserved; see [preservation-after.json](aw01c/preservation-after.json). No changes were made to the other project repositories.

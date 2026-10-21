from airfield_aircraft_vision.generalization_dataset import (
    rank_training_tiles,
    choose_training_groups,
)
import pytest


def test_training_rank_preserves_every_id_and_prioritizes_small_and_crowded():
    images = [dict(id=i, file_name=f"x{i}.png") for i in range(5)]
    anns = [dict(image_id=i, bbox=[0, 0, 40, 40]) for i in range(5)]
    anns += [dict(image_id=3, bbox=[0, 0, 10, 10]) for _ in range(5)]
    anns += [dict(image_id=4, bbox=[0, 0, 2, 2])]
    ranked = rank_training_tiles(images, anns)
    assert ranked[0]["id"] == 4
    assert ranked[1]["id"] == 3
    assert {i["id"] for i in ranked} == set(range(5))
    assert ranked == rank_training_tiles(images, anns)


def test_group_expansion_preserves_groups_and_prefers_new_country():
    rows = [dict(loc_id=str(i), Country=c) for i, c in enumerate(["A", "A", "B", "C"])]
    groups = {str(i): f"g{i}" for i in range(4)}
    selected = choose_training_groups({"g1", "g2", "g3"}, {"g0"}, rows, groups, total=3)
    assert selected == {"g0", "g2", "g3"}


def test_group_expansion_refuses_reduction_or_shortage():
    with pytest.raises(ValueError):
        choose_training_groups(set(), {"g"}, [], {}, total=0)
    with pytest.raises(ValueError):
        choose_training_groups(set(), set(), [], {}, total=1)


def _cached_inputs(tmp_path):
    import json
    from airfield_aircraft_vision.preprocessing import sha256_file

    root = tmp_path / "isolated"
    root.mkdir()
    base = tmp_path / "baseline.json"
    legacy = tmp_path / "legacy.json"
    base.write_text("baseline")
    legacy.write_text("legacy")
    reservation = dict(
        aw01b_manifest_sha256=sha256_file(base),
        legacy_manifest_sha256=sha256_file(legacy),
    )
    (root / "test-reservation.json").write_text(json.dumps(reservation))
    request = dict(
        baseline_sha256=sha256_file(base),
        reservation_sha256=sha256_file(root / "test-reservation.json"),
        total_train_groups=32,
        train_tiles=16,
        seed=41,
        max_mib=192,
        workers=4,
    )
    (root / "request.json").write_text(json.dumps(request))
    (root / "protocol.json").write_text("{}")
    (root / "manifest.json").write_text("{}")
    return base, legacy, root


@pytest.mark.parametrize("missing", ["request.json", "protocol.json"])
def test_cached_expansion_requires_frozen_configuration(tmp_path, missing):
    from airfield_aircraft_vision.generalization_dataset import prepare_generalization

    base, legacy, root = _cached_inputs(tmp_path)
    (root / missing).unlink()
    with pytest.raises(ValueError, match="request or protocol"):
        prepare_generalization(base, legacy, root)


def test_cached_expansion_rejects_protocol_tamper(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import airfield_aircraft_vision.generalization_dataset as module

    base, legacy, root = _cached_inputs(tmp_path)
    cached = SimpleNamespace(preparation=dict(protocol_sha256="0" * 64))
    monkeypatch.setattr(module, "load_manifest", lambda _: cached)
    with pytest.raises(ValueError, match="protocol checksum"):
        module.prepare_generalization(base, legacy, root)


def test_cached_expansion_rejects_metadata_tamper(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from airfield_aircraft_vision.preprocessing import sha256_file
    import airfield_aircraft_vision.generalization_dataset as module

    base, legacy, root = _cached_inputs(tmp_path)
    (root / "metadata").mkdir()
    (root / "metadata" / "source.csv").write_text("changed")
    cached = SimpleNamespace(
        preparation=dict(
            protocol_sha256=sha256_file(root / "protocol.json"),
            metadata_sha256={"source.csv": "0" * 64},
        )
    )
    monkeypatch.setattr(module, "load_manifest", lambda _: cached)
    with pytest.raises(ValueError, match="metadata changed"):
        module.prepare_generalization(base, legacy, root)


def test_exact_cached_expansion_is_read_only_without_network(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from airfield_aircraft_vision.preprocessing import sha256_file
    import airfield_aircraft_vision.generalization_dataset as module

    base, legacy, root = _cached_inputs(tmp_path)
    cached = SimpleNamespace(
        preparation=dict(
            protocol_sha256=sha256_file(root / "protocol.json"), metadata_sha256={}
        )
    )
    monkeypatch.setattr(module, "load_manifest", lambda _: cached)

    def forbidden(*a, **k):
        raise AssertionError("must not download cached data")

    monkeypatch.setattr(module, "BoundedDownload", forbidden)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert module.prepare_generalization(base, legacy, root) is cached
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


@pytest.mark.parametrize("changed", ["request", "baseline"])
def test_expansion_rejects_changed_lineage_or_request(tmp_path, changed):
    import json
    from airfield_aircraft_vision.generalization_dataset import prepare_generalization

    base, legacy, root = _cached_inputs(tmp_path)
    if changed == "baseline":
        base.write_text("different")
    else:
        config = json.loads((root / "request.json").read_text())
        config["seed"] = 42
        (root / "request.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="differs"):
        prepare_generalization(base, legacy, root)


def _rich_fixture():
    index = {
        "images": [{"id": 1, "file_name": "a.png"}],
        "annotations": [{"id": 4, "image_id": 1, "bbox": [0, 0, 10, 20]}],
    }
    rich = {
        "images": list(index["images"]),
        "annotations": [dict(index["annotations"][0], partialDec=0.5)],
        "categories": [{"id": 100, "name": "irrelevant attribute"}],
    }
    return index, rich


def test_rich_contract_joins_by_id_and_box_without_using_categories():
    from airfield_aircraft_vision.generalization_dataset import validate_rich_contract

    validate_rich_contract(*_rich_fixture())


@pytest.mark.parametrize("change", ["image", "bbox", "id", "duplicate", "visibility"])
def test_rich_contract_rejects_changed_annotation_identity_or_visibility(change):
    from airfield_aircraft_vision.generalization_dataset import validate_rich_contract

    index, rich = _rich_fixture()
    if change == "image":
        rich["images"] = [{"id": 1, "file_name": "other.png"}]
    elif change == "bbox":
        rich["annotations"][0]["bbox"] = [1, 0, 10, 20]
    elif change == "id":
        rich["annotations"][0]["id"] = 5
    elif change == "duplicate":
        rich["annotations"].append(dict(rich["annotations"][0]))
    else:
        rich["annotations"][0]["partialDec"] = float("nan")
    with pytest.raises(ValueError):
        validate_rich_contract(index, rich)

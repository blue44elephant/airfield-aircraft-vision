"""AW-01C native-pixel, resource-bounded training with validation-only selection."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from collections import defaultdict, deque
from pathlib import Path

from .architecture_backend import initialize_architecture_from_coco, RESNET_ARCHITECTURE
from .torch_backend import ARCHITECTURE, TorchDetector
from .dataset import load_manifest, resolve_image
from .preprocessing import NativeRaster, sha256_file
from .improved_training import augment_native, collect_predictions
from .experiment_metrics import choose_threshold
from .generalization_metrics import evaluate_generalization, reviewed_negative_ids
from .workflows import assert_holdout


def sample_order(samples, seed):
    """Seeded airfield round robin; every sample appears exactly once per epoch."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.source.airfield_group].append(sample)
    keys = sorted(groups)
    rng.shuffle(keys)
    queues = {}
    for key in keys:
        rng.shuffle(groups[key])
        queues[key] = deque(groups[key])
    order = []
    while any(queues.values()):
        for key in keys:
            if queues[key]:
                order.append(queues[key].popleft())
    return order


def validate_configuration(architecture, epochs, lr, threads, max_seconds):
    if architecture not in (ARCHITECTURE, RESNET_ARCHITECTURE):
        raise ValueError("unsupported architecture")
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (epochs, threads))
        or not 1 <= epochs <= 8
        or not 1 <= threads <= 8
    ):
        raise ValueError("bounded epoch/thread configuration required")
    if not math.isfinite(lr) or not 0 < lr <= 0.01:
        raise ValueError("finite learning rate in (0,.01] required")
    if not math.isfinite(max_seconds) or not 30 <= max_seconds <= 3600:
        raise ValueError("training time limit must be 30..3600 seconds")


def validate_training_sources(samples):
    """Preserve VISION-0 physical-resolution and quality routing before training."""
    from .capability import SceneMetadata, route_scene, CapabilityTier

    for sample in samples:
        source = sample.source
        decision = route_scene(
            SceneMetadata(
                sensor_domain=source.sensor_domain,
                native_gsd_m=source.native_gsd_m,
                panchromatic_gsd_m=source.panchromatic_gsd_m,
                pan_sharpened_gsd_m=source.pan_sharpened_gsd_m,
                effective_spatial_resolution_m=source.effective_resolution_m,
                image_quality=source.quality,
            )
        )
        if source.sensor_domain != "optical" or decision.tier != CapabilityTier.A:
            raise ValueError("training requires Tier A optical imagery")


def train_generalization(
    manifest_path,
    initial_weights,
    outdir,
    *,
    architecture=RESNET_ARCHITECTURE,
    epochs=3,
    lr=0.0025,
    threads=4,
    max_seconds=1500,
    seed=41,
    initial_checkpoint=None,
):
    import torch
    import torchvision
    import psutil

    validate_configuration(architecture, epochs, lr, threads, max_seconds)
    manifest_path, initial_weights, outdir = map(
        Path, (manifest_path, initial_weights, outdir)
    )
    if outdir.exists():
        raise ValueError("fresh isolated training output directory required")
    manifest = load_manifest(manifest_path)
    train = [s for s in manifest.samples if s.split == "train"]
    val = [s for s in manifest.samples if s.split == "validation"]
    if not train or not val:
        raise ValueError("nonempty training and validation required")
    if any(
        s.source.sensor_domain != "optical" or s.source.native_gsd_m > 1 for s in train
    ):
        raise ValueError("native high-resolution optical training only")
    validate_training_sources(train)
    if any(s.source.width > 512 or s.source.height > 512 for s in train):
        raise ValueError("native training tiles must be <=512 pixels")
    # Ignored annotations must never silently become negative training labels.
    if any(s.ignored_boxes for s in train):
        raise ValueError("training tiles with ignored regions need explicit exclusion")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    if initial_checkpoint is None:
        model = initialize_architecture_from_coco(initial_weights, architecture)
        initial_sha = sha256_file(initial_weights)
    else:
        initial = TorchDetector(Path(initial_checkpoint), candidate_floor=0.005)
        if initial.manifest["architecture"] != architecture:
            raise ValueError("initial checkpoint architecture differs")
        assert_holdout(
            [s for s in manifest.samples if s.split in ("validation", "test")],
            initial.manifest,
        )
        model = initial.model
        initial_sha = sha256_file(Path(initial_checkpoint))
        del initial
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    model.roi_heads.score_thresh = 0.005
    # Identical resource limits for both arms; evaluation retains official proposal defaults.
    model.rpn._pre_nms_top_n["training"] = 1000
    model.rpn._post_nms_top_n["training"] = 512
    model.roi_heads.fg_bg_sampler.batch_size_per_image = 128
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    config = dict(
        architecture=architecture,
        epochs=epochs,
        lr=lr,
        optimizer="SGD",
        momentum=0.9,
        weight_decay=0.0005,
        lr_schedule="multiply .3 after each epoch",
        backbone="fully frozen incl FPN",
        batch_size=1,
        threads=threads,
        max_seconds=max_seconds,
        seed=seed,
        native_tile_size=512,
        augmentation="D4 flips/90-degree rotations; no resize",
        training_rpn_pre_nms=1000,
        training_rpn_post_nms=512,
        roi_batch=128,
        sampler="seeded airfield round robin; each image once",
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint else None,
    )
    info = dict(
        architecture=architecture,
        preprocessing="native-rgb-v1",
        candidate_score_floor=0.005,
        experimental=True,
        experimental_input_bounds_m=[0.2, 1.0],
        classes=["background", "aircraft"],
        training_manifest_sha256=sha256_file(manifest_path),
        initial_weights_sha256=initial_sha,
        training_airfields=sorted({s.source.airfield_group for s in train}),
        training_captures=sorted({s.source.capture_group for s in train}),
        training_source_checksums=sorted({s.source.sha256 for s in train}),
        validation_airfields=sorted({s.source.airfield_group for s in val}),
        configuration=config,
        source_code_sha256={
            p.name: sha256_file(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("architecture_backend.py"),
                Path(__file__).with_name("torch_backend.py"),
            )
        },
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        limitations=[
            "Experimental detector; confidence not a calibrated probability.",
            "Reviewed local negative crops do not establish full-scene false-alarm rates.",
        ],
    )
    assert_holdout(val, info)
    outdir.mkdir(parents=True)
    process = psutil.Process()
    start = time.perf_counter()
    cpu_start = process.cpu_times()
    peak = process.memory_info().rss
    (outdir / "run.json").write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                configuration=config,
                manifest_sha256=sha256_file(manifest_path),
                initial_weights_sha256=initial_sha,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    detector = object.__new__(TorchDetector)
    detector.device = torch.device("cpu")
    detector.manifest = info
    detector.model = model
    history = []
    best_rank = None
    stop_reason = "epoch_limit"
    for epoch in range(1, epochs + 1):
        model.train()
        model.backbone.eval()
        losses = []
        rng = random.Random(seed + epoch)
        order = sample_order(train, seed + epoch)
        for step, sample in enumerate(order, 1):
            if (outdir / "STOP").exists():
                stop_reason = "cancellation"
                break
            if time.perf_counter() - start >= max_seconds:
                stop_reason = "time_limit"
                break
            with NativeRaster(
                resolve_image(manifest_path, sample), sample.source
            ) as raster:
                pixels, valid = raster.read(
                    (0, 0, sample.source.width, sample.source.height), 512
                )
            if not valid.any():
                raise ValueError("all-nodata training image")
            pixels, boxes = augment_native(
                pixels,
                [b.xyxy for b in sample.boxes],
                rng.randrange(4),
                bool(rng.randrange(2)),
            )
            image = torch.from_numpy(pixels).float() / 255.0
            target = dict(
                boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                labels=torch.ones(len(boxes), dtype=torch.int64),
            )
            values = model([image], [target])
            loss = sum(values.values())
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            memory = process.memory_info()
            peak = max(peak, memory.rss, getattr(memory, "peak_wset", 0))
            if step % 25 == 0 or step == len(order):
                progress = dict(
                    epoch=epoch,
                    step=step,
                    total_steps=len(order),
                    elapsed_seconds=time.perf_counter() - start,
                    mean_loss=sum(losses) / len(losses),
                    rss_mib=process.memory_info().rss / 1024**2,
                )
                (outdir / "progress.json").write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )
                print(json.dumps(progress), flush=True)
        if not losses:
            break
        model.eval()
        _, validation, predictions = collect_predictions(
            manifest_path, detector, "validation"
        )
        selection = choose_threshold(validation, predictions, split="validation")
        apron_ids, background_ids = reviewed_negative_ids(
            manifest.preparation, validation
        )
        evaluation = evaluate_generalization(
            validation,
            predictions,
            selection["selected_threshold"],
            geography_by_scene=manifest.preparation.get("geography_by_scene"),
            weather_by_scene=manifest.preparation.get("metadata_weather_by_scene"),
            verified_empty_image_ids=apron_ids,
            verified_background_image_ids=background_ids,
        )
        row = dict(
            epoch=epoch,
            steps=len(losses),
            full_epoch=len(losses) == len(train),
            mean_training_loss=sum(losses) / len(losses),
            learning_rate=optimizer.param_groups[0]["lr"],
            validation=selection,
            validation_ap50=evaluation["metrics"]["ap50"],
            elapsed_seconds=time.perf_counter() - start,
            peak_rss_mib=peak / 1024**2,
        )
        history.append(row)
        rank = (
            selection["metrics"]["f1"] or 0,
            -selection["metrics"]["count_mae"],
            selection["selected_threshold"],
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            payload_info = dict(
                info,
                selected_epoch=epoch,
                selected_threshold=selection["selected_threshold"],
                selected_validation_metrics=evaluation["metrics"],
                selection_split="validation",
            )
            torch.save(
                dict(state_dict=model.state_dict(), manifest=payload_info),
                outdir / "best.pt",
            )
            (outdir / "best-validation.json").write_text(
                json.dumps(evaluation, indent=2), encoding="utf-8"
            )
        (outdir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                dict(
                    epoch=epoch,
                    validation=evaluation["metrics"],
                    threshold=selection["selected_threshold"],
                    elapsed=row["elapsed_seconds"],
                )
            ),
            flush=True,
        )
        scheduler.step()
        if stop_reason != "epoch_limit":
            break
        if (
            len(history) >= 3
            and max(h["validation"]["metrics"]["f1"] or 0 for h in history[-2:])
            < best_rank[0]
        ):
            stop_reason = "validation_no_improvement_two_epochs"
            break
    cpu_end = process.cpu_times()
    resources = dict(
        stop_reason=stop_reason,
        elapsed_seconds=time.perf_counter() - start,
        peak_rss_mib=max(peak, getattr(process.memory_info(), "peak_wset", 0))
        / 1024**2,
        cpu_user_seconds=cpu_end.user - cpu_start.user,
        cpu_system_seconds=cpu_end.system - cpu_start.system,
        threads=threads,
        completed_epochs=sum(r["full_epoch"] for r in history),
        training_tiles=len(train),
        validation_tiles=len(val),
    )
    (outdir / "resources.json").write_text(
        json.dumps(resources, indent=2), encoding="utf-8"
    )
    if not history:
        raise ValueError("training stopped before any update; no selected model")
    return history

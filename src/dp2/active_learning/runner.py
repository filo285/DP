"""Recoverable active-learning study with immutable acquisition checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd
import torch

from dp2.common.uncertainty import SCORING_PROTOCOL
from dp2.common.training import save_json as save_clean_json
from .acquisition import select_images
from .annotation import reveal_selection
from .artifacts import local_path, read_json, stage, study_lock, verify_stage
from .evaluation import EVALUATION_PROTOCOL, evaluate_predictor
from .scoring import export_pool_scores
from .state import load_state, resolve_path, save_json, seed_for, sha256, write_state
from .training import load_predictor, state_provenance, train_subset


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def elapsed(start):
    return (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds()


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def predictor_manifest(path, method):
    return Path(path) / ("model.json" if method == "mc_dropout" else "ensemble.json")


def score_stage(experiment, state_path, predictor, output, device, sequential):
    state, _, pool, _ = load_state(experiment, state_path)
    models, cfg, manifest = load_predictor(predictor, state_provenance(experiment, state), device, sequential)
    export_pool_scores(output, pool, models, cfg, manifest, experiment.data_root,
                       experiment.config["mc_passes"], experiment.config["seed"],
                       state["repetition"], state["round"], sequential)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def selection_stage(experiment, state_path, method, strategy, scores_path, output):
    state, _, pool, _ = load_state(experiment, state_path)
    scores = None
    source = None
    if scores_path is not None:
        source = read_json(scores_path / "report.json")
        if (source["provenance"] != state_provenance(experiment, state) or source["method"] != method
                or source["scores_sha256"] != sha256(scores_path / "pool_scores.csv")):
            raise ValueError("Pool scores mismatch current predictor/state")
        scores = pd.read_csv(scores_path / "pool_scores.csv", dtype={"image_id": str})
    seed = seed_for(experiment.config["seed"], "acquisition", state["repetition"], state["round"], strategy)
    selected, ranking = select_images(pool.image_id, strategy, experiment.config["acquisition_size"], seed, scores)
    output.mkdir(parents=True)
    selected.to_csv(output / "acquisition.csv", index=False)
    ranking.to_csv(output / "ranking.csv", index=False)
    _selection_preview(experiment, pool, selected, output / "selected_images.png")
    save_json(output / "selection.json", {"status": "complete", "method": method,
              "strategy": strategy, "seed": seed, "selected_count": len(selected),
              "provenance": state_provenance(experiment, state),
              "fallback_count": int(selected.fallback.sum()),
              "invalid_score_fraction": float((~ranking.score_valid).mean()) if scores is not None else None,
              "acquisition_sha256": sha256(output / "acquisition.csv"),
              "score_report_sha256": sha256(scores_path / "report.json") if source else None})


def _selection_preview(experiment, pool, selected, destination):
    """Show selected inputs only; reference masks are never opened here."""
    import cv2
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    shown = selected.head(6)
    fig, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 3.5), squeeze=False, layout="constrained")
    source = pool.set_index("image_id")
    for ax, row in zip(axes[0], shown.itertuples()):
        path = resolve_path(experiment.data_root, source.loc[row.image_id, "image_path"])
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(path)
        scale = min(1., 512 / max(raw.shape[:2]))
        raw = cv2.resize(raw, (max(1, int(raw.shape[1] * scale)), max(1, int(raw.shape[0] * scale))))
        ax.imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        score = f"{row.score:.5g}" if np.isfinite(row.score) else "no valid score"
        ax.set_title(f"{row.image_id}\n{score}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Selected images — input images only", fontsize=11)
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def _record(output, name, record):
    output.mkdir(parents=True)
    save_clean_json(output / name, record)


def run_study(experiment, output, methods=None, strategies=None, repetitions=None,
              device="cpu", sequential=False, smoke=False, resume=False, hook=None):
    """Run the declared grid, then matched full-pool references; never open test masks."""
    root = Path(output).resolve()
    methods = list(methods or experiment.config["methods"])
    strategies = list(strategies or experiment.config["strategies"])
    repetitions = list(range(experiment.config["repetitions"])) if repetitions is None else list(repetitions)
    for values, allowed in ((methods, experiment.config["methods"]), (strategies, experiment.config["strategies"]),
                            (repetitions, range(experiment.config["repetitions"]))):
        if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
            raise ValueError("Invalid study grid")
    smoke = bool(smoke or experiment.config.get("smoke", False))
    design = {"version": 1, "experiment_sha256": experiment.digest, "methods": methods,
              "strategies": strategies, "repetitions": repetitions, "smoke": smoke,
              "evaluation_protocol": EVALUATION_PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
              "config": experiment.config, "full_pool_references": True}
    if root.exists() and not resume:
        raise FileExistsError("Study exists; use --resume with the same frozen design")
    with study_lock(root):
        if (root / "study.json").exists():
            if read_json(root / "study.json")["design"] != design:
                raise ValueError("Resume settings differ from the frozen study design")
        else:
            save_json(root / "study.json", {"design": design, "started_utc": utc_now()})
        if (root / "frozen.json").exists():
            verify_completed_study(root, experiment)
            return root
        try:
            arms, references = [], []
            for repetition in repetitions:
                for method in methods:
                    for strategy in strategies:
                        arms.append(_run_arm(experiment, root, repetition, method, strategy,
                                             device, sequential, smoke, hook))
                    references.append(_run_reference(experiment, root, repetition, method, device, sequential, smoke))
            save_json(root / "study_complete.json", {"status": "complete", "study_sha256": sha256(root / "study.json"),
                       "arms": arms, "references": references})
        except BaseException as error:
            save_json(root / "last_error.json", {"time_utc": utc_now(), "error": str(error), "type": type(error).__name__})
            raise
    return root


def _run_arm(experiment, root, repetition, method, strategy, device, sequential, smoke, hook):
    arm = root / f"repeat_{repetition:02d}" / method / strategy
    arm.mkdir(parents=True, exist_ok=True)
    if not (arm / "arm.json").exists():
        save_json(arm / "arm.json", {"started_utc": utc_now(), "method": method,
                                   "strategy": strategy, "repetition": repetition})
    arm_info = read_json(arm / "arm.json")
    initial = experiment.directory / f"repeat_{repetition:02d}" / "initial"
    state_path = stage(arm / "round_00" / "state", lambda p: shutil.copytree(initial, p))
    records, cumulative_steps = [], 0
    for round_index in range(experiment.config["acquisitions"] + 1):
        round_dir = arm / f"round_{round_index:02d}"
        state, labeled, pool, _ = load_state(experiment, state_path)
        expected_count = experiment.config["initial_size"] + round_index * experiment.config["acquisition_size"]
        if state["round"] != round_index or len(labeled) != expected_count:
            raise ValueError("Round population disagrees with the frozen schedule")
        save_json(arm / "progress.json", {"round": round_index, "stage": "training"})
        shared = root / "shared" / f"repeat_{repetition:02d}" / method if round_index == 0 else round_dir
        predictor = stage(shared / "training", lambda p: train_subset(experiment, state_path, method, p,
                          device, smoke, resume=True), resumable=True)
        manifest_path = predictor_manifest(predictor, method)
        manifest = read_json(manifest_path)
        if manifest["provenance"] != state_provenance(experiment, state):
            raise ValueError("Cached predictor belongs to a different labeled population")
        cumulative_steps += sum(m["optimizer_steps"] for m in manifest["members"])
        if hook:
            hook("trained", round_index)
        validation = stage(shared / "validation", lambda p: evaluate_predictor(
            experiment, state_path, predictor, p, "val", device, sequential))
        metrics = read_json(validation / "report.json")["metrics"]
        record = {"round": round_index, "method": method, "strategy": strategy, "repetition": repetition,
                  "labeled_count": len(labeled), "pool_count": len(pool), "smoke": manifest["smoke"],
                  "state": relative(root, state_path), "predictor": relative(root, predictor),
                  "validation": relative(root, validation), "metrics": metrics,
                  "elapsed_wall_seconds": elapsed(arm_info["started_utc"]),
                  "cumulative_optimizer_steps": cumulative_steps,
                  "training_seconds": sum(m["elapsed_seconds"] for m in manifest["members"]),
                  "selected_epochs": [m["best_epoch"] for m in manifest["members"]],
                  "completed_epochs": [m["completed_epochs"] for m in manifest["members"]]}
        save_json(arm / "progress.json", {"round": round_index, "stage": "evaluated"})
        terminal = round_index == experiment.config["acquisitions"]
        if not terminal:
            scores_path = None
            if strategy != "random":
                scores_path = stage(shared / "scores", lambda p: score_stage(experiment, state_path, predictor, p, device, sequential))
            selection = stage(round_dir / "selection", lambda p: selection_stage(experiment, state_path, method, strategy, scores_path, p))
            selected = pd.read_csv(selection / "acquisition.csv", dtype={"image_id": str})
            selection_info = read_json(selection / "selection.json")
            event_id = hashlib.sha256(json.dumps([sha256(root / "study.json"), repetition,
                                                 method, strategy, round_index]).encode()).hexdigest()
            event = {"event_id": event_id, "event_type": "acquisition", "experiment_sha256": experiment.digest,
                     "method": method, "strategy": strategy, "repetition": repetition,
                     "source_round": round_index, "destination_round": round_index + 1,
                     "requested_utc": utc_now(), "elapsed_wall_seconds": elapsed(arm_info["started_utc"]),
                     "labeled_before": len(labeled), "pool_before": len(pool),
                     "requested_count": experiment.config["acquisition_size"], "selected_count": len(selected),
                     "labeled_after": len(labeled) + len(selected), "pool_after": len(pool) - len(selected),
                     "selected": selected.to_dict("records"), "fallback_count": selection_info["fallback_count"],
                     "invalid_score_fraction": selection_info["invalid_score_fraction"],
                     "selected_dataset_counts": pool[pool.image_id.isin(selected.image_id)].dataset.value_counts().to_dict(),
                     "predictor": relative(root, predictor), "predictor_manifest_sha256": sha256(manifest_path),
                     "members": manifest["members"], "state_sha256": sha256(state_path / "state.json"),
                     "labeled_ids_sha256": state["labeled_ids_sha256"], "pool_ids_sha256": state["pool_ids_sha256"],
                     "selection": relative(root, selection), "selection_sha256": sha256(selection / "acquisition.csv"),
                     "validation": relative(root, validation), "validation_metrics": metrics,
                     "validation_sha256": sha256(validation / "report.json"),
                     "scoring_protocol": SCORING_PROTOCOL, "acquisition_seed": selection_info["seed"],
                     "cumulative_optimizer_steps": cumulative_steps, "smoke": manifest["smoke"]}
            event_dir = stage(round_dir / "event", lambda p: _record(p, "acquisition_checkpoint.json", event))
            saved_event = read_json(event_dir / "acquisition_checkpoint.json")
            if saved_event["selection_sha256"] != sha256(selection / "acquisition.csv"):
                raise ValueError("Previously published acquisition differs from saved selection")
            if hook:
                hook("selected", round_index)
            next_state = stage(arm / f"round_{round_index + 1:02d}" / "state", lambda p: reveal_selection(
                experiment, state_path, selected.image_id, p, smoke=manifest["smoke"],
                acquisition={"event_id": saved_event["event_id"], "event_sha256": sha256(event_dir / "acquisition_checkpoint.json")}))
            if hook:
                hook("committed", round_index)
            record.update(event=relative(root, event_dir), next_state=relative(root, next_state),
                          elapsed_wall_seconds=saved_event["elapsed_wall_seconds"],
                          score_seconds=read_json(scores_path / "report.json")["elapsed_seconds"] if scores_path else 0.)
        else:
            record.update(event=None, next_state=None, score_seconds=0., terminal_reason="annotation_budget_reached")
        record_dir = stage(round_dir / "record", lambda p: _record(p, "round.json", record))
        records.append(relative(root, record_dir))
        if not terminal:
            state_path = next_state
    stage(arm / "complete", lambda p: _record(p, "arm_complete.json", {"status": "complete", "records": records}))
    save_json(arm / "progress.json", {"round": experiment.config["acquisitions"], "stage": "complete"})
    return relative(root, arm / "complete")


def _run_reference(experiment, root, repetition, method, device, sequential, smoke):
    folder = root / "references" / f"repeat_{repetition:02d}" / method
    train = experiment.splits["train"]
    state_path = stage(folder / "state", lambda p: write_state(p, experiment, train,
                       train.iloc[:0].copy(), experiment.annotations.reveal(train.image_id), repetition, 0, smoke=smoke))
    predictor = stage(folder / "training", lambda p: train_subset(experiment, state_path, method, p, device, smoke, resume=True), resumable=True)
    validation = stage(folder / "validation", lambda p: evaluate_predictor(experiment, state_path, predictor, p, "val", device, sequential))
    record = {"role": "full_pool_reference", "method": method, "repetition": repetition, "smoke": smoke,
              "state": relative(root, state_path), "predictor": relative(root, predictor),
              "validation": relative(root, validation), "labeled_count": len(train)}
    stage(folder / "record", lambda p: _record(p, "reference.json", record))
    return relative(root, folder / "record")


def verify_completed_study(root, experiment):
    root = Path(root)
    study = read_json(root / "study.json")
    completion = read_json(root / "study_complete.json")
    if (study["design"]["experiment_sha256"] != experiment.digest or
            completion["study_sha256"] != sha256(root / "study.json") or completion["status"] != "complete"):
        raise ValueError("Study is incomplete or belongs to another experiment")
    design = study["design"]
    if design["evaluation_protocol"] != EVALUATION_PROTOCOL or design["scoring_protocol"] != SCORING_PROTOCOL:
        raise ValueError("Study protocol differs from the current evaluation/scoring implementation")
    expected_arms = {f"repeat_{r:02d}/{m}/{s}/complete" for r in design["repetitions"]
                     for m in design["methods"] for s in design["strategies"]}
    expected_references = {f"references/repeat_{r:02d}/{m}/record" for r in design["repetitions"] for m in design["methods"]}
    if (set(completion["arms"]) != expected_arms or len(completion["arms"]) != len(expected_arms)
            or set(completion["references"]) != expected_references
            or len(completion["references"]) != len(expected_references)):
        raise ValueError("Completion manifest does not cover the declared study grid")
    records = []
    stages = set()
    for arm in completion["arms"]:
        folder = local_path(root, arm)
        verify_stage(folder)
        arm_records = read_json(folder / "arm_complete.json")["records"]
        expected_records = [str(Path(arm).parent / f"round_{i:02d}" / "record").replace("\\", "/")
                            for i in range(design["config"]["acquisitions"] + 1)]
        if arm_records != expected_records:
            raise ValueError("An arm has incomplete or out-of-order rounds")
        for record_path in arm_records:
            record_folder = local_path(root, record_path)
            verify_stage(record_folder)
            record = read_json(record_folder / "round.json")
            state, _, _, _ = load_state(experiment, local_path(root, record["state"]))
            predictor = read_json(predictor_manifest(local_path(root, record["predictor"]), record["method"]))
            if predictor["provenance"] != state_provenance(experiment, state):
                raise ValueError("Round predictor/state provenance mismatch")
            records.append((record_path, record))
            for key in ("state", "predictor", "validation", "event", "next_state"):
                if record.get(key):
                    stages.add(record[key])
            if record.get("event"):
                event = read_json(local_path(root, record["event"]) / "acquisition_checkpoint.json")
                stages.add(event["selection"])
    for path in completion["references"]:
        verify_stage(local_path(root, path))
        record = read_json(local_path(root, path) / "reference.json")
        records.append((path, record))
        stages.update(record[key] for key in ("state", "predictor", "validation"))
    for path in sorted(stages):
        verify_stage(local_path(root, path))
    return study, records


def freeze_study(experiment, root):
    root = Path(root)
    with study_lock(root):
        _, records = verify_completed_study(root, experiment)
        snapshot = {"study.json": sha256(root / "study.json"), "study_complete.json": sha256(root / "study_complete.json")}
        for path, record in records:
            snapshot[path + "/_complete.json"] = sha256(local_path(root, path) / "_complete.json")
            for key in ("state", "predictor", "validation", "event"):
                if record.get(key):
                    name = record[key] + "/_complete.json"
                    snapshot[name] = sha256(local_path(root, name))
        frozen = root / "frozen.json"
        if frozen.exists():
            if read_json(frozen)["snapshot"] != snapshot:
                raise ValueError("Frozen study has changed")
        else:
            save_json(frozen, {"frozen_utc": utc_now(), "snapshot": snapshot,
                               "evaluation_protocol": EVALUATION_PROTOCOL, "experiment_sha256": experiment.digest})
    return frozen


def evaluate_frozen_study(experiment, root, device="cpu", sequential=False):
    root = Path(root)
    with study_lock(root):
        frozen = read_json(root / "frozen.json")  # Fails before accessing any test labels.
        for name, digest in frozen["snapshot"].items():
            if sha256(local_path(root, name)) != digest:
                raise ValueError("Frozen study changed; test evaluation refused")
        _, records = verify_completed_study(root, experiment)
        results = []
        for record_path, record in records:
            state_path, predictor = local_path(root, record["state"]), local_path(root, record["predictor"])
            key = sha256(predictor_manifest(predictor, record["method"]))
            destination = root / "test_evaluation" / key
            stage(destination, lambda p: evaluate_predictor(experiment, state_path, predictor, p,
                                                            "test", device, sequential, test_authorized=True))
            results.append({"record": record_path, "evaluation": relative(root, destination)})
        save_json(root / "test_evaluation.json", {"status": "complete", "frozen_sha256": sha256(root / "frozen.json"), "results": results})
    return root / "test_evaluation.json"

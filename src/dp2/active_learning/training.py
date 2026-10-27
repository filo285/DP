"""Train and load predictors tied to an exact labeled-set snapshot."""
from dataclasses import asdict, fields
import json
from pathlib import Path

import torch

from dp2.baseline import Config, build_model
from dp2.common.training import train_model
from .data import supervised_frame
from .state import ids_hash, load_state, save_json, seed_for, sha256


def state_provenance(experiment, state):
    return {"experiment_sha256": experiment.digest,
            "global_split_sha256": experiment.protocol["split_sha256"],
            "labeled_ids_sha256": state["labeled_ids_sha256"],
            "pool_ids_sha256": state["pool_ids_sha256"],
            "repetition": state["repetition"], "round": state["round"]}


def train_subset(experiment, state_directory, method, output, device="cpu", smoke=False, resume=False):
    if method not in experiment.config["methods"]:
        raise ValueError("Unknown uncertainty method")
    state, labeled, _, annotations = load_state(experiment, state_directory)
    smoke = smoke or experiment.config.get("smoke", False) or state.get("smoke", False)
    training = dict(experiment.config["training"])
    if smoke:
        training.update(img_size=128, epochs=1, batch_size=2, encoder_weights=None)
    if training.get("encoder_weights") == "none":
        training["encoder_weights"] = None
    cfg = Config(**training, device=device, csv_paths={})
    train = supervised_frame(labeled, annotations, experiment.data_root)
    val_public = experiment.splits["val"].head(4) if smoke else experiment.splits["val"]
    val_annotations = experiment.private_annotations("val")
    val_annotations = val_annotations[val_annotations.image_id.isin(val_public.image_id)]
    val = supervised_frame(val_public, val_annotations, experiment.data_root)
    provenance = state_provenance(experiment, state)
    output = Path(output)
    manifest_path = output / ("model.json" if method == "mc_dropout" else "ensemble.json")
    if output.exists() and resume and not manifest_path.exists():
        from .artifacts import archive_partial
        archive_partial(output)
    existing = output.exists()
    if existing and not resume:
        raise FileExistsError(output)
    if not existing:
        output.mkdir(parents=True)
        (output / "splits").mkdir()
        for name, frame in (("train", train), ("val", val), ("test", experiment.splits["test"])):
            frame.to_csv(output / "splits" / f"{name}.csv", index=False)
    hashes = {p.name: sha256(p) for p in (output / "splits").glob("*.csv")}
    count = 1 if method == "mc_dropout" else (2 if smoke else experiment.config["ensemble_members"])
    seeds = [seed_for(experiment.config["seed"], "training", state["repetition"], state["round"], i)
             for i in range(count)]
    if len(set(seeds)) != count:
        raise ValueError("Member seed collision")
    manifest = {"version": 1, "method": method, "status": "training", "scope": "active_learning",
                "smoke": smoke, "provenance": provenance, "split_sha256": hashes,
                "validation_ids_sha256": ids_hash(val_public.image_id),
                "initialization": "fresh", "config": asdict(cfg), "members": [],
                "requested_members": count, "torch_version": str(torch.__version__),
                "device": device, "selection": "best mean validation disc/cup Dice"}
    if existing:
        previous = json.loads(manifest_path.read_text())
        for key in ("method", "scope", "smoke", "provenance", "split_sha256", "config", "requested_members"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Cannot resume changed training inputs: {key}")
        manifest = previous
    else:
        save_json(manifest_path, manifest)
    for index, seed in enumerate(seeds):
        member_dir = output / f"member_{index:02d}_seed_{seed}"
        if member_dir.exists() and (member_dir / "training.json").exists():
            result = json.loads((member_dir / "training.json").read_text())
            if (result["seed"] != seed or result["provenance"] != provenance
                    or result.get("checkpoint_sha256") != sha256(member_dir / "best.pth")
                    or result.get("history_sha256") != sha256(member_dir / "history.json")):
                raise ValueError("Completed member integrity failure during resume")
        else:
            if member_dir.exists():
                from .artifacts import archive_partial
                archive_partial(member_dir)
            result = train_model(train, val, cfg, seed, member_dir, hashes, method, smoke, provenance)
        checkpoint = member_dir / "best.pth"
        result.pop("checkpoint")
        entry = {**result, "checkpoint": checkpoint.relative_to(output).as_posix(), "sha256": sha256(checkpoint)}
        if index < len(manifest["members"]):
            if manifest["members"][index] != entry:
                raise ValueError("Completed member manifest changed")
        else:
            manifest["members"].append(entry)
        save_json(manifest_path, manifest)
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)
    return manifest


def load_predictor(directory, expected_provenance, device="cpu", sequential=False):
    """Reject historical/full-pool checkpoints and mismatched active-learning states."""
    directory = Path(directory)
    candidates = [directory / name for name in ("model.json", "ensemble.json") if (directory / name).exists()]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one active-learning predictor manifest")
    manifest = json.loads(candidates[0].read_text())
    if (manifest.get("status") != "complete" or manifest.get("scope") != "active_learning"
            or manifest.get("initialization") != "fresh"
            or manifest.get("provenance") != expected_provenance):
        raise ValueError("Predictor is incomplete, full-pool, or has mismatched labeled-set provenance")
    method, members = manifest["method"], manifest["members"]
    if (method not in ("mc_dropout", "deep_ensembles") or len(members) != manifest["requested_members"]
            or (method == "mc_dropout" and len(members) != 1)
            or (method == "deep_ensembles" and len(members) < 2)
            or len({m["seed"] for m in members}) != len(members)):
        raise ValueError("Invalid member count or duplicate seeds")
    for name, digest in manifest["split_sha256"].items():
        if name not in ("train.csv", "val.csv", "test.csv") or sha256(directory / "splits" / name) != digest:
            raise ValueError("Training split snapshot changed")
    models, seen = [], set()
    for member in members:
        path = (directory / member["checkpoint"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint must be inside its predictor directory")
        digest = sha256(path)
        if digest != member["sha256"] or digest in seen:
            raise ValueError("Corrupt or duplicate member checkpoint")
        seen.add(digest)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("provenance") != expected_provenance
                or checkpoint.get("initialization") != "fresh"
                or checkpoint.get("seed") != member["seed"]
                or checkpoint.get("split_sha256") != manifest["split_sha256"]
                or checkpoint.get("method") != method or checkpoint.get("smoke") != manifest["smoke"]):
            raise ValueError("Checkpoint provenance mismatch")
        raw = checkpoint["cfg"]
        if any(raw.get(k) != v for k, v in manifest["config"].items() if k != "seed"):
            raise ValueError("Checkpoint configuration mismatch")
        cfg = Config(**{k: v for k, v in raw.items() if k in {f.name for f in fields(Config)}})
        cfg.encoder_weights = None  # Loading must not download or initialize pretrained weights.
        cfg.device = device
        model = build_model(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model.to("cpu" if sequential and method == "deep_ensembles" else device).eval())
    manifest["directory"] = str(directory.resolve())
    manifest["manifest_sha256"] = sha256(candidates[0])
    return models, cfg, manifest

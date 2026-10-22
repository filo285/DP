"""Load ensemble members with compatible preprocessing and training provenance."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import torch
from dp2.baseline import Config, build_model


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_ensemble(directory, device="cpu"):
    directory = Path(directory)
    manifest = json.loads((directory / "ensemble.json").read_text())
    members = manifest["members"]
    if manifest.get("status") != "complete" or len(members) < 2:
        raise ValueError("Ensemble training is incomplete or has fewer than two members")
    if len({m["seed"] for m in members}) != len(members):
        raise ValueError("Member seeds must be distinct")
    models, hashes, signatures = [], set(), set()
    cfg = None
    for member in members:
        path = (directory / member["checkpoint"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Checkpoint must be inside the ensemble directory")
        digest = sha256(path)
        if digest != member["sha256"] or digest in hashes:
            raise ValueError("Checkpoint hash mismatch or duplicate checkpoint")
        hashes.add(digest)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("seed") != member["seed"] or checkpoint.get("split_sha256") != manifest["split_sha256"]:
            raise ValueError("Checkpoint seed or split provenance mismatch")
        raw = checkpoint["cfg"]
        signature = tuple(raw[k] for k in ("encoder", "img_size", "dropout", "col_image", "col_mask"))
        signatures.add(signature)
        if len(signatures) != 1:
            raise ValueError("Incompatible member architecture or preprocessing")
        cfg = Config(**{k: v for k, v in raw.items() if k in {f.name for f in fields(Config)}})
        cfg.encoder_weights, cfg.device = None, device
        model = build_model(cfg)
        model.load_state_dict(checkpoint["model"], strict=True)
        models.append(model.to(device).eval())
    return models, cfg, manifest

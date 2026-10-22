"""Verify DP1 validation parity, then export MC-dropout maps and image scores."""
import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dp2.baseline import (Config, FundusSegDataset, build_model, build_transforms,
                          evaluate, load_master_tables, make_splits,
                          _binary_disc_cup, dice_binary)
from dp2.mc_dropout.inference import mc_dropout_predict
from dp2.common.export import export_predictions
from dp2.common.uncertainty import SCORING_PROTOCOL


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passes", type=int, default=30)
    parser.add_argument("--limit", type=int, default=8, help="validation images for UQ; 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8, help="deterministic parity batch size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/mc_dropout/pilot")
    args = parser.parse_args()
    if args.passes < 2 or args.limit < 0 or args.batch_size < 1:
        parser.error("passes >= 2, limit >= 0 and batch-size >= 1 required")
    args.output.mkdir(parents=True, exist_ok=False)
    paths = json.loads((ROOT / "configs/paths.json").read_text())
    data_root = Path(paths["dp1_root"])
    checkpoint_path = data_root / paths["baseline_checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = Config(**{k: v for k, v in checkpoint["cfg"].items() if k in {f.name for f in fields(Config)}})
    cfg.device, cfg.encoder_weights = args.device, None
    cfg.csv_paths = {k: str(data_root / v) for k, v in cfg.csv_paths.items()}
    for dataset in ("origa", "refuge"):
        cfg.csv_paths[dataset] = str(data_root / paths[f"{dataset}_table"])
        if not Path(cfg.csv_paths[dataset]).is_file():
            raise FileNotFoundError(cfg.csv_paths[dataset])
    splits = make_splits(load_master_tables(cfg), cfg)
    manifest_dir = args.output / "splits"
    manifest_dir.mkdir()
    seen = set()
    for name, frame in zip(("train", "val", "test"), splits):
        if frame.empty:
            raise ValueError(f"Empty {name} split")
        ids = frame[cfg.col_image].map(lambda x: str(Path(x).resolve()).casefold())
        if ids.duplicated().any() or seen.intersection(ids):
            raise ValueError("Duplicate image paths within/across splits")
        seen.update(ids)
        frame.to_csv(manifest_dir / f"{name}.csv", index=False)
    _, transform = build_transforms(cfg)
    validation = FundusSegDataset(splits[1], transform, cfg)
    model = build_model(cfg).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Checking deterministic parity on all {len(validation)} validation images...", flush=True)
    metrics = evaluate(model, DataLoader(validation, batch_size=args.batch_size), cfg)
    reference = checkpoint["val_metrics"]
    differences = {k: abs(metrics[k] - reference[k]) for k in reference}
    parity = all(np.isfinite(v) and v < 1e-5 for v in differences.values())
    metadata = {
        "scoring_protocol": SCORING_PROTOCOL,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "epoch": checkpoint["epoch"], "checkpoint_config": checkpoint["cfg"],
        "torch_version": str(torch.__version__), "device": args.device,
        "method": "mc_dropout", "passes": args.passes, "seed": args.seed, "image_seed_rule": "seed + image index", "validation_metrics": metrics,
        "stored_validation_metrics": reference, "absolute_differences": differences,
        "parity_passed": parity, "parity_tolerance": 1e-5,
        "split_sha256": {p.name: sha256(p) for p in manifest_dir.glob("*.csv")},
        "scope": "Full-pool checkpoint UQ validation pilot; not a limited-label AL experiment",
        "roi": "predicted foreground intersect valid pixels only; empty foreground gives NaN, excluded from ranking",
        "std": "population standard deviation of per-class softmax probabilities; image score averages classes",
        "entropy_units": "nats", "uq_batch_size": 1,
        "geometry": "512 square DP1 letterboxed coordinates; valid mask excludes padding",
    }
    report = args.output / "report.json"
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Validation metrics:", metrics, "Parity:", parity, flush=True)
    if not parity:
        raise RuntimeError("Validation parity failed; inspect report.json before UQ experiments")
    count = export_predictions(validation, transform, cfg,
                               lambda image, index, valid: mc_dropout_predict(model, image, args.passes, args.seed + index, valid_mask=valid),
                               args.output, args.limit, "mc_dropout")
    metadata["uq_images"] = count
    metadata["status"] = "complete"
    report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()

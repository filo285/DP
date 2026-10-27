"""Train independent U-Nets sequentially on identical frozen dataset splits."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import torch
from dp2.baseline import Config, load_master_tables, make_splits
from dp2.common.training import train_model, segmentation_loss, seed_training, save_json
from dp2.deep_ensembles.checkpoints import sha256



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/deep_ensembles/full")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="2 members, 1 epoch, 8 train/4 val images at 128px; NOT research results")
    args = parser.parse_args()
    if args.members < 2 or args.epochs < 1 or args.batch_size < 1 or args.seed < 0 or args.seed + args.members >= 2**32:
        parser.error("members >= 2, epochs and batch-size >= 1, valid nonnegative 32-bit seeds required")
    if args.smoke:
        args.members, args.epochs, args.batch_size, args.encoder_weights = 2, 1, 2, "none"
    cfg = Config(epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                 encoder_weights=None if args.encoder_weights == "none" else "imagenet",
                 img_size=128 if args.smoke else 512)
    paths = json.loads((ROOT / "configs/paths.json").read_text())
    data_root = Path(paths["dp1_root"])
    cfg.csv_paths = {k: str(data_root / v) for k, v in cfg.csv_paths.items()}
    for name in ("origa", "refuge"):
        cfg.csv_paths[name] = str(data_root / paths[f"{name}_table"])
        if not Path(cfg.csv_paths[name]).is_file():
            raise FileNotFoundError(cfg.csv_paths[name])
    splits = list(make_splits(load_master_tables(cfg), cfg))
    seen = set()
    for frame in splits:
        ids = frame[cfg.col_image].map(lambda p: str(Path(p).resolve()).casefold())
        if frame.empty or ids.duplicated().any() or seen.intersection(ids):
            raise ValueError("Empty split or overlapping/duplicate image paths")
        seen.update(ids)
    if args.smoke:
        splits[0], splits[1] = splits[0].head(8), splits[1].head(4)
    args.output.mkdir(parents=True, exist_ok=False)
    split_dir = args.output / "splits"
    split_dir.mkdir()
    for name, frame in zip(("train", "val", "test"), splits):
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    hashes = {p.name: sha256(p) for p in split_dir.glob("*.csv")}
    manifest = {"method": "deep_ensembles", "status": "training", "smoke": args.smoke,
                "scope": "smoke_only" if args.smoke else "full_pool_supervised",
                "members": [], "requested_members": args.members, "split_sha256": hashes,
                "torch_version": str(torch.__version__), "device": args.device,
                "initialization": "fresh model per seed; optional common ImageNet encoder; no DP1 checkpoint",
                "selection": "best mean validation disc/cup Dice; patience 10",
                "config": asdict(cfg)}
    manifest_path = args.output / "ensemble.json"
    save_json(manifest_path, manifest)
    for index in range(args.members):
        seed = args.seed + index
        member_dir = args.output / f"member_{index:02d}_seed_{seed}"
        result = train_model(splits[0], splits[1], cfg, seed, member_dir, hashes,
                             method="deep_ensembles", smoke=args.smoke)
        checkpoint_path = Path(result["checkpoint"])
        manifest["members"].append({"seed": seed,
                                    "checkpoint": checkpoint_path.relative_to(args.output).as_posix(),
                                    "sha256": sha256(checkpoint_path), "best_mean_dice": result["best_mean_dice"]})
        save_json(manifest_path, manifest)
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)
    print("Ensemble saved:", args.output, flush=True)


if __name__ == "__main__":
    main()

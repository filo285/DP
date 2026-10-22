"""Train independent U-Nets sequentially on identical frozen dataset splits."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dp2.baseline import (Config, FundusSegDataset, build_model, build_transforms,
                          evaluate, load_master_tables, make_splits)
from dp2.deep_ensembles.checkpoints import sha256


def segmentation_loss(logits, target):
    """Same CE + soft multiclass Dice objective as DP1, including background."""
    probs = logits.softmax(1)
    onehot = torch.nn.functional.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probs * onehot).sum((0, 2, 3))
    cardinality = (probs + onehot).sum((0, 2, 3))
    dice = (2 * intersection + 1e-6) / (cardinality + 1e-6)
    return nn.functional.cross_entropy(logits, target) + 1 - dice.mean()


def seed_training(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


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
        seed_training(seed)
        cfg.seed = seed
        member_dir = args.output / f"member_{index:02d}_seed_{seed}"
        member_dir.mkdir()
        train_tf, eval_tf = build_transforms(cfg)
        train_tf.set_random_seed(seed)  # Albumentations has an independent RNG.
        train_ds = FundusSegDataset(splits[0], train_tf, cfg)
        val_ds = FundusSegDataset(splits[1], eval_tf, cfg)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed),
                                  num_workers=0, drop_last=len(train_ds) >= cfg.batch_size)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)
        model = build_model(cfg).to(cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        amp = cfg.amp and torch.device(cfg.device).type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best, patience, history = -1., 0, []
        checkpoint_path = member_dir / "best.pth"
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            total, samples = 0., 0
            for step, (images, targets) in enumerate(train_loader, 1):
                images, targets = images.to(cfg.device), targets.to(cfg.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(torch.device(cfg.device).type, enabled=amp):
                    loss = segmentation_loss(model(images), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(images)
                samples += len(images)
                if step % 20 == 0:
                    print(f"Member {index+1}/{args.members}, epoch {epoch}, batch {step}/{len(train_loader)}", flush=True)
            scheduler.step()
            metrics = evaluate(model, val_loader, cfg)
            score = (metrics["dice_disc"] + metrics["dice_cup"]) / 2
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite validation Dice")
            history.append({"epoch": epoch, "loss": total / samples, **metrics})
            save_json(member_dir / "history.json", history)
            print(f"Member {index+1}/{args.members}, epoch {epoch}: loss={total/samples:.4f}, val Dice={score:.4f}", flush=True)
            if score > best:
                best, patience = score, 0
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch,
                            "val_metrics": metrics, "seed": seed, "split_sha256": hashes,
                            "method": "deep_ensembles", "smoke": args.smoke}, checkpoint_path)
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break
        manifest["members"].append({"seed": seed,
                                    "checkpoint": checkpoint_path.relative_to(args.output).as_posix(),
                                    "sha256": sha256(checkpoint_path), "best_mean_dice": best})
        save_json(manifest_path, manifest)
        del model, optimizer, scheduler, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)
    print("Ensemble saved:", args.output, flush=True)


if __name__ == "__main__":
    main()

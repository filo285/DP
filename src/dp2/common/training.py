"""Shared fresh-model trainer operating only on explicit supervised manifests."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time
import hashlib
import os
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dp2.baseline import FundusSegDataset, build_model, build_transforms, evaluate


def segmentation_loss(logits, target):
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
    # Undefined secondary validation metrics are encoded as JSON null.
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, list):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(value), indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_model(train_frame, val_frame, cfg, seed, output, split_hashes,
                method="deep_ensembles", smoke=False, provenance=None):
    """Always build a fresh model/optimizer; no checkpoint initialization option."""
    if (train_frame.empty or val_frame.empty or cfg.epochs < 1 or cfg.batch_size < 1
            or cfg.early_stop_patience < 1 or not 0 <= seed < 2**32):
        raise ValueError("Invalid training subset/configuration/seed")
    for frame in (train_frame, val_frame):
        paths = frame[cfg.col_image].map(lambda p: str(Path(p).resolve()).casefold())
        if paths.duplicated().any():
            raise ValueError("Duplicate supervised image paths")
    train_paths = {str(Path(p).resolve()).casefold() for p in train_frame[cfg.col_image]}
    val_paths = {str(Path(p).resolve()).casefold() for p in val_frame[cfg.col_image]}
    if train_paths & val_paths:
        raise ValueError("Training and validation overlap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = replace(cfg, seed=seed)
    seed_training(seed)
    train_tf, eval_tf = build_transforms(cfg)
    train_tf.set_random_seed(seed)
    train_ds = FundusSegDataset(train_frame, train_tf, cfg)
    val_ds = FundusSegDataset(val_frame, eval_tf, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0,
                              drop_last=len(train_ds) >= cfg.batch_size)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=0)
    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    amp = cfg.amp and torch.device(cfg.device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, patience, history, steps, best_epoch = -1., 0, [], 0, 0
    start = time.monotonic()
    checkpoint = output / "best.pth"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, samples = 0., 0
        for images, targets in train_loader:
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
            steps += 1
        scheduler.step()
        metrics = evaluate(model, val_loader, cfg)
        score = (metrics["dice_disc"] + metrics["dice_cup"]) / 2
        if not np.isfinite(score):
            raise RuntimeError("Nonfinite validation Dice")
        history.append({"epoch": epoch, "optimizer_steps": steps, "loss": total / samples,
                        "elapsed_seconds": time.monotonic() - start, **metrics})
        save_json(output / "history.json", history)
        print(f"Seed {seed}, epoch {epoch}: loss={total/samples:.4f}, val Dice={score:.4f}", flush=True)
        if score > best:
            best, patience, best_epoch = score, 0, epoch
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "epoch": epoch,
                        "optimizer_steps": steps, "val_metrics": metrics, "seed": seed,
                        "split_sha256": split_hashes, "method": method, "smoke": smoke,
                        "provenance": provenance,
                        "initialization": "fresh", "encoder_initialization": cfg.encoder_weights}, checkpoint)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break
    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with (output / "history.json").open("rb") as stream:
        history_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
               "history_sha256": history_hash, "seed": seed, "best_mean_dice": best,
               "best_epoch": best_epoch, "completed_epochs": epoch, "optimizer_steps": steps,
               "elapsed_seconds": time.monotonic() - start, "provenance": provenance}
    save_json(output / "training.json", {**summary, "checkpoint": "best.pth"})
    del model, optimizer, scheduler, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary

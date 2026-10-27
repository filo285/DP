"""Compact mask-free pool scoring with existing BALD/vCDR definitions."""
from pathlib import Path
import time

import pandas as pd
import torch

from dp2.common.uncertainty import SCORING_PROTOCOL
from dp2.deep_ensembles.inference import ensemble_predict
from dp2.mc_dropout.inference import mc_dropout_predict
from .acquisition import SCORE_COLUMNS
from .data import ImagePoolDataset
from .state import ids_hash, save_json, seed_for, sha256


def score_pool(pool, models, cfg, method, data_root, passes=30, seed=42,
               repetition=0, round_index=0, sequential=False):
    if method not in ("mc_dropout", "deep_ensembles"):
        raise ValueError("Unknown uncertainty method")
    if (method == "mc_dropout" and len(models) != 1) or (method == "deep_ensembles" and len(models) < 2):
        raise ValueError("Incorrect model count")
    dataset = ImagePoolDataset(pool, cfg, data_root)
    rows = []
    for index in range(len(dataset)):
        image_id, image, valid = dataset[index]
        images, valid = image[None].to(cfg.device), valid[None].to(cfg.device)
        inference_seed = seed_for(seed, "inference", repetition, round_index, image_id)
        if method == "mc_dropout":
            result = mc_dropout_predict(models[0], images, passes, inference_seed, valid_mask=valid)
        else:
            result = ensemble_predict(models, images, valid_mask=valid, sequential=sequential)
        stats = {**result.image_scores(), **result.vcdr_scores()}
        rows.append({"image_id": image_id, "method": method,
                     "inference_seed": inference_seed if method == "mc_dropout" else "",
                     **{key: stats[key][0].item() for key in SCORE_COLUMNS[3:]}})
        if (index + 1) % 25 == 0:
            print(f"Scored {index + 1}/{len(dataset)} pool images", flush=True)
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


def export_pool_scores(output, pool, models, cfg, manifest, data_root, passes, seed,
                       repetition, round_index, sequential=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "method": manifest["method"], "smoke": manifest["smoke"],
              "provenance": manifest["provenance"], "members": manifest["members"],
              "predictor_directory": manifest["directory"],
              "predictor_manifest_sha256": manifest["manifest_sha256"],
              "pool_ids_sha256": ids_hash(pool.image_id), "pool_count": len(pool),
              "scoring_protocol": SCORING_PROTOCOL, "mc_passes": passes,
              "seed": seed, "repetition": repetition, "round": round_index,
              "sequential_ensemble": sequential, "device": cfg.device,
              "seed_rule": "SHA256(base, inference, repetition, round, stable image ID)",
              "torch_version": str(torch.__version__)}
    save_json(output / "report.json", report)
    started = time.monotonic()
    try:
        frame = score_pool(pool, models, cfg, manifest["method"], data_root, passes,
                           seed, repetition, round_index, sequential)
        frame.to_csv(output / "pool_scores.csv", index=False)
        report.update(status="complete", elapsed_seconds=time.monotonic() - started,
                      scores_sha256=sha256(output / "pool_scores.csv"),
                      invalid_bald_count=int((~frame.image_uncertainty_valid.astype(bool)).sum()),
                      invalid_vcdr_count=int((~frame.vcdr_uncertainty_valid.astype(bool)).sum()))
    except Exception as error:
        report.update(status="failed", error=str(error))
        save_json(output / "report.json", report)
        raise
    save_json(output / "report.json", report)
    return frame

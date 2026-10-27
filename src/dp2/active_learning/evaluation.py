"""Method-specific evaluation, including padding-aware metrics and failure coverage."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from dp2.baseline import (FundusSegDataset, build_transforms, dice_binary,
                          hausdorff_binary, vcdr)
from dp2.common.training import save_json
from dp2.deep_ensembles.inference import ensemble_predict
from dp2.mc_dropout.inference import mc_dropout_predict
from .data import ImagePoolDataset, supervised_frame
from .state import ids_hash, load_state, seed_for, sha256
from .training import load_predictor, state_provenance

EVALUATION_PROTOCOL = {
    "version": 1, "prediction": "argmax(mean softmax probabilities)",
    "primary": "valid_mean_dice", "vcdr": "ratio of final mean-probability segmentation",
    "geometry": "letterboxed input coordinates; Hausdorff in resized-image pixels",
    "invalid_vcdr": "conditional MAE plus valid count and missing-disc counts; no imputation",
    "curve_area": "trapezoidal mean disc/cup Dice over common label interval, divided by interval width",
    "target": None,
}


def image_metrics(prediction, reference, valid):
    result = {}
    for prefix, keep in (("legacy", np.ones_like(valid, dtype=bool)), ("valid", valid)):
        pd_disc, pd_cup = (prediction > 0) & keep, (prediction == 2) & keep
        gt_disc, gt_cup = (reference > 0) & keep, (reference == 2) & keep
        result.update({f"{prefix}_dice_disc": float(dice_binary(pd_disc, gt_disc)),
                       f"{prefix}_dice_cup": float(dice_binary(pd_cup, gt_cup)),
                       f"{prefix}_hd_disc": float(hausdorff_binary(pd_disc, gt_disc)),
                       f"{prefix}_hd_cup": float(hausdorff_binary(pd_cup, gt_cup))})
        pred_ratio, gt_ratio = vcdr(pd_disc, pd_cup), vcdr(gt_disc, gt_cup)
        result[f"{prefix}_vcdr_absolute_error"] = float(abs(pred_ratio - gt_ratio))
        result[f"{prefix}_mean_dice"] = (result[f"{prefix}_dice_disc"] + result[f"{prefix}_dice_cup"]) / 2
        if prefix == "valid":
            result.update(vcdr_prediction=float(pred_ratio), vcdr_reference=float(gt_ratio),
                          missing_disc=not bool(pd_disc.any()), missing_cup=not bool(pd_cup.any()),
                          missing_reference_disc=not bool(gt_disc.any()))
    return result


def aggregate_metrics(frame):
    result = {"images": len(frame)}
    for name in frame.columns:
        if name.startswith(("legacy_", "valid_")):
            values = frame[name].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            result[name] = float(finite.mean()) if len(finite) else None
            result[name + "_count"] = len(finite)
    for name in ("missing_disc", "missing_cup", "missing_reference_disc"):
        result[name + "_count"] = int(frame[name].sum())
    result["vcdr_valid_count"] = int(np.isfinite(frame.valid_vcdr_absolute_error).sum())
    result["vcdr_coverage"] = result["vcdr_valid_count"] / len(frame)
    return result


def evaluate_predictor(experiment, state_directory, predictor, output, split="val",
                       device="cpu", sequential=False, test_authorized=False):
    if split not in ("val", "test") or (split == "test" and not test_authorized):
        raise ValueError("Test evaluation requires the separate frozen-study evaluation job")
    state, _, _, _ = load_state(experiment, state_directory)
    models, cfg, manifest = load_predictor(predictor, state_provenance(experiment, state), device, sequential)
    public = experiment.splits[split]
    if manifest["smoke"]:
        public = public.head(4)
    annotations = experiment.private_annotations(split)
    annotations = annotations[annotations.image_id.isin(public.image_id)]
    supervised = supervised_frame(public, annotations, experiment.data_root)
    _, transform = build_transforms(cfg)
    targets = FundusSegDataset(supervised, transform, cfg)
    images = ImagePoolDataset(public, cfg, experiment.data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    for index in range(len(images)):
        image_id, image, valid = images[index]
        _, reference = targets[index]
        inputs, keep = image[None].to(device), valid[None].to(device)
        seed = seed_for(experiment.config["seed"], "evaluation", split, state["repetition"], state["round"], image_id)
        if manifest["method"] == "mc_dropout":
            result = mc_dropout_predict(models[0], inputs, experiment.config["mc_passes"], seed, valid_mask=keep)
        else:
            result = ensemble_predict(models, inputs, valid_mask=keep, sequential=sequential)
        prediction = result.prediction[0].cpu().numpy()
        rows.append({"image_id": image_id, "inference_seed": seed,
                     **image_metrics(prediction, reference.numpy(), valid.numpy())})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_image.csv", index=False)
    report = {"status": "complete", "split": split, "method": manifest["method"],
              "smoke": manifest["smoke"], "provenance": manifest["provenance"],
              "predictor_manifest_sha256": manifest["manifest_sha256"],
              "members": manifest["members"], "evaluation_protocol": EVALUATION_PROTOCOL,
              "evaluation_ids_sha256": ids_hash(public.image_id),
              "mc_passes": experiment.config["mc_passes"], "member_count": len(models),
              "metrics": aggregate_metrics(frame), "per_image_sha256": sha256(output / "per_image.csv"),
              "elapsed_seconds": time.monotonic() - started, "device": device,
              "gpu": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None}
    save_json(output / "report.json", report)
    del models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report

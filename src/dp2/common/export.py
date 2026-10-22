"""Shared export format for fair comparison of UQ methods."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache/matplotlib"))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dp2.baseline import _binary_disc_cup, dice_binary
from dp2.common.vcdr import summarize_vcdr, vertical_cup_to_disc


def export_predictions(validation, transform, cfg, predict, output, limit, method):
    count = min(limit or len(validation), len(validation))
    rows, sample_rows = [], []
    for index in range(count):
        image, target = validation[index]
        source = validation.df.iloc[index]
        raw = cv2.imread(source[cfg.col_image])
        valid = transform(image=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                          mask=np.ones(raw.shape[:2], dtype=np.uint8))["mask"].bool()
        result = predict(image[None].to(cfg.device), index, valid[None].to(cfg.device))
        scores = {**result.image_scores(), **result.vcdr_scores()}
        reference_vcdr = vertical_cup_to_disc(target[None], valid[None])[0][0].item()
        prediction = result.prediction[0].cpu().numpy()
        pd_disc, pd_cup = _binary_disc_cup(prediction)
        gt_disc, gt_cup = _binary_disc_cup(target.numpy())
        row = {"index": index, "image_path": source[cfg.col_image], "method": method,
               **{k: v[0].item() for k, v in scores.items()},
               "dice_disc": dice_binary(pd_disc, gt_disc), "dice_cup": dice_binary(pd_cup, gt_cup),
               "vcdr_reference": reference_vcdr,
               "vcdr_absolute_error": abs(scores["vcdr_prediction"][0].item() - reference_vcdr)}
        rows.append(row)
        for sample_index in range(len(result.vcdr_samples)):
            prefix = summarize_vcdr(result.vcdr_samples[:sample_index + 1])
            sample_rows.append({"index": index, "image_path": source[cfg.col_image], "method": method,
                                "sample_index": sample_index, "iteration": sample_index + 1,
                                "sample_kind": "dropout_pass" if method == "mc_dropout" else "ensemble_member",
                                "vcdr": result.vcdr_samples[sample_index, 0].item(),
                                "disc_height_pixels": result.disc_heights[sample_index, 0].item(),
                                "cup_height_pixels": result.cup_heights[sample_index, 0].item(),
                                "running_vcdr_mean": prefix["vcdr_mean"][0].item(),
                                "running_vcdr_uncertainty": prefix["vcdr_uncertainty"][0].item(),
                                "running_uncertainty_valid": prefix["vcdr_uncertainty_valid"][0].item()})
        arrays = {name: getattr(result, name)[0].cpu().numpy() for name in
                  ("mean_probabilities", "std_probabilities", "predictive_entropy",
                   "expected_entropy", "mutual_information")}
        np.savez_compressed(output / f"image_{index:04d}.npz", **arrays,
                            prediction=prediction, valid_mask=valid.numpy(),
                            vcdr_samples=result.vcdr_samples[:, 0].cpu().numpy(),
                            disc_heights=result.disc_heights[:, 0].cpu().numpy(),
                            cup_heights=result.cup_heights[:, 0].cpu().numpy())
        if index < 8:
            rgb = (image.permute(1, 2, 0).numpy() * np.array([.229, .224, .225])
                   + np.array([.485, .456, .406])).clip(0, 1)
            fig, axes = plt.subplots(1, 5, figsize=(18, 4), layout="constrained")
            for ax, value, title in zip(axes, (rgb, prediction, arrays["std_probabilities"].mean(0),
                                              arrays["predictive_entropy"], arrays["mutual_information"]),
                                        ("Input", f"{method} segmentation", "Mean class std", "Predictive entropy", "Mutual information")):
                im = ax.imshow(value, cmap="magma" if ax in axes[2:] else "viridis")
                ax.set_title(title)
                ax.axis("off")
                if ax in axes[2:]:
                    fig.colorbar(im, ax=ax, shrink=.7)
            fig.suptitle(f"Image uncertainty (mean BALD): {row['image_uncertainty']:.5f} | "
                         f"vCDR mean: {row['vcdr_mean']:.4f} | vCDR uncertainty (std): {row['vcdr_uncertainty']:.5f}")
            fig.savefig(output / f"image_{index:04d}.png", dpi=130)
            plt.close(fig)
        print(f"UQ {index + 1}/{count}: BALD={row['bald']:.6f}, vCDR std={row['vcdr_uncertainty']:.6f}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "image_scores.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(output / "vcdr_iterations.csv", index=False)
    # Stable ties follow manifest order; masks/quality metrics never enter ranking.
    frame[frame.image_uncertainty_valid].sort_values("image_uncertainty", ascending=False, kind="stable").to_csv(output / "ranking_bald.csv", index=False)
    frame[~frame.image_uncertainty_valid].to_csv(output / "image_uncertainty_invalid.csv", index=False)
    eligible = frame[frame.vcdr_uncertainty_valid]
    eligible.sort_values("vcdr_uncertainty", ascending=False, kind="stable").to_csv(output / "ranking_vcdr.csv", index=False)
    frame[~frame.vcdr_uncertainty_valid].to_csv(output / "vcdr_invalid.csv", index=False)
    return count

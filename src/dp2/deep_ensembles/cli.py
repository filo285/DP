"""Export validation uncertainty from a completed independent ensemble."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import pandas as pd
import torch
from dp2.baseline import FundusSegDataset, build_transforms
from dp2.common.export import export_predictions
from dp2.common.uncertainty import SCORING_PROTOCOL
from dp2.deep_ensembles.checkpoints import load_ensemble, sha256
from dp2.deep_ensembles.inference import ensemble_predict


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "runs/deep_ensembles/predictions")
    parser.add_argument("--limit", type=int, default=8, help="0 = all validation images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be >= 0")
    models, cfg, manifest = load_ensemble(args.ensemble, args.device)
    split_path = args.ensemble / "splits/val.csv"
    if sha256(split_path) != manifest["split_sha256"]["val.csv"]:
        raise ValueError("Validation manifest changed since training")
    validation_frame = pd.read_csv(split_path)
    if validation_frame.empty:
        raise ValueError("Validation split is empty")
    _, transform = build_transforms(cfg)
    validation = FundusSegDataset(validation_frame, transform, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scoring_protocol": SCORING_PROTOCOL, "method": "deep_ensembles", "status": "running", "ensemble": str(args.ensemble.resolve()),
              "members": manifest["members"], "member_count": len(models), "smoke": manifest["smoke"],
              "scope": manifest["scope"], "split_sha256": manifest["split_sha256"],
              "device": args.device, "torch_version": str(torch.__version__), "img_size": cfg.img_size,
              "dropout_at_inference": False, "entropy_units": "nats",
              "bald": "ensemble mutual information H(mean p)-mean H(p); BALD-style disagreement, not exact Bayesian posterior",
              "roi": "predicted foreground intersect valid pixels only; empty foreground gives NaN, excluded from ranking"}
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["uq_images"] = export_predictions(validation, transform, cfg,
                                             lambda images, index, valid: ensemble_predict(models, images, valid_mask=valid),
                                             args.output, args.limit, "deep_ensembles")
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()

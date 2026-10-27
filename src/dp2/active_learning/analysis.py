"""Summarize immutable results and plot acquisition markers without inference."""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import local_path, read_json, verify_stage
from .runner import verify_completed_study
from .state import save_json, sha256

METRICS = ("valid_mean_dice", "valid_dice_disc", "valid_dice_cup", "valid_vcdr_absolute_error",
           "vcdr_coverage", "valid_hd_disc", "valid_hd_cup", "missing_disc_count",
           "legacy_mean_dice", "legacy_dice_disc", "legacy_dice_cup", "legacy_vcdr_absolute_error")


def curve_area(labels, values):
    labels, values = np.asarray(labels, dtype=float), np.asarray(values, dtype=float)
    if (len(labels) < 2 or len(labels) != len(values) or not np.isfinite(values).all()
            or not np.isfinite(labels).all() or not (np.diff(labels) > 0).all()):
        raise ValueError("Curve area requires finite values at strictly increasing label budgets")
    return float(np.trapezoid(values, labels) / (labels[-1] - labels[0]))


def summarize_study(experiment, root, output, split="val", include_smoke=False):
    root, output = Path(root), Path(output)
    study, records = verify_completed_study(root, experiment)
    design = study["design"]
    if design["smoke"] and not include_smoke:
        raise ValueError("Smoke runs are excluded; use --include-smoke for a labeled functional-check summary")
    if split not in ("val", "test"):
        raise ValueError("Unknown evaluation split")
    test_results = {}
    if split == "test":
        frozen = read_json(root / "frozen.json")
        for name, digest in frozen["snapshot"].items():
            if sha256(local_path(root, name)) != digest:
                raise ValueError("Frozen study has changed")
        index = read_json(root / "test_evaluation.json")
        if index["status"] != "complete" or index["frozen_sha256"] != sha256(root / "frozen.json"):
            raise ValueError("Final test evaluation missing or stale")
        test_results = {row["record"]: row["evaluation"] for row in index["results"]}
        if len(test_results) != len(index["results"]) or set(test_results) != {path for path, _ in records}:
            raise ValueError("Test evaluations do not cover the frozen study exactly once")
    rows, references, events = [], [], []
    for record_path, record in records:
        eval_path = record["validation"] if split == "val" else test_results[record_path]
        verify_stage(local_path(root, eval_path))
        report = read_json(local_path(root, eval_path) / "report.json")
        if report["split"] != split or report["method"] != record["method"]:
            raise ValueError("Evaluation identity mismatch")
        if report["smoke"] and not include_smoke:
            raise ValueError("Unexpected smoke artifact in research summary")
        row = {"method": record["method"], "repetition": record["repetition"], "split": split,
               "labeled_count": record["labeled_count"], "smoke": report["smoke"],
               "predictor": record["predictor"], "evaluation": eval_path,
               "predictor_manifest_sha256": report["predictor_manifest_sha256"], **report["metrics"]}
        row.update(device=report["device"], gpu=report["gpu"])
        if record.get("role") == "full_pool_reference":
            references.append(row)
            continue
        row.update(strategy=record["strategy"], round=record["round"],
                   elapsed_wall_seconds=record["elapsed_wall_seconds"],
                   cumulative_optimizer_steps=record["cumulative_optimizer_steps"],
                   training_seconds=record["training_seconds"], score_seconds=record["score_seconds"],
                   selected_epochs=json.dumps(record["selected_epochs"]),
                   completed_epochs=json.dumps(record["completed_epochs"]),
                   point_type="initial" if record["round"] == 0 else ("terminal" if not record["event"] else "intermediate"),
                   requests_more_images=bool(record["event"]))
        rows.append(row)
        if record["event"]:
            event = read_json(local_path(root, record["event"]) / "acquisition_checkpoint.json")
            # Anchor the checkpoint to its persisted request, even after interrupted publication.
            row["elapsed_wall_seconds"] = event["elapsed_wall_seconds"]
            events.append({"event_id": event["event_id"], "method": record["method"], "strategy": record["strategy"],
                           "repetition": record["repetition"], "source_round": record["round"],
                           "destination_round": event["destination_round"], "requested_utc": event["requested_utc"],
                           "elapsed_wall_seconds": event["elapsed_wall_seconds"], "labeled_before": event["labeled_before"],
                           "labeled_after": event["labeled_after"], "selected_count": event["selected_count"],
                           "fallback_count": event["fallback_count"], "invalid_score_fraction": event["invalid_score_fraction"],
                           "selected_ids": json.dumps([r["image_id"] for r in event["selected"]]),
                           "members": json.dumps(event["members"]), "predictor": event["predictor"],
                           "checkpoint_record": record["event"] + "/acquisition_checkpoint.json",
                           **{f"selected_{k}": v for k, v in event["selected_dataset_counts"].items()}})
    curves, reference_frame, event_frame = pd.DataFrame(rows), pd.DataFrame(references), pd.DataFrame(events)
    if not event_frame.empty and event_frame.event_id.duplicated().any():
        raise ValueError("Duplicate acquisition event IDs")
    keys = ["method", "strategy", "repetition"]
    schedule = [design["config"]["initial_size"] + i * design["config"]["acquisition_size"]
                for i in range(design["config"]["acquisitions"] + 1)]
    areas = []
    for identity, frame in curves.groupby(keys):
        frame = frame.sort_values("round")
        if frame.labeled_count.tolist() != schedule:
            raise ValueError("Incomplete or mismatched learning-curve budgets")
        areas.append(dict(zip(keys, identity), normalized_dice_auc=curve_area(frame.labeled_count, frame.valid_mean_dice)))
    area_frame = pd.DataFrame(areas)
    for index, event in event_frame.iterrows():
        matching = curves[(curves.method == event.method) & (curves.strategy == event.strategy)
                          & (curves.repetition == event.repetition)]
        for label, round_index in (("before", event.source_round), ("after", event.destination_round)):
            match = matching[matching["round"] == round_index].iloc[0]
            for metric in METRICS:
                event_frame.loc[index, f"{label}_{metric}"] = match.get(metric)
    random_areas = area_frame[area_frame.strategy == "random"][["method", "repetition", "normalized_dice_auc"]]
    paired = area_frame.merge(random_areas.rename(columns={"normalized_dice_auc": "random_auc"}), on=["method", "repetition"], how="left")
    paired["difference_from_random"] = paired.normalized_dice_auc - paired.random_auc
    paired_summary = paired.groupby(["method", "strategy"])[["normalized_dice_auc", "difference_from_random"]].agg(["mean", "std", "count"])
    paired_summary.columns = ["_".join(parts) for parts in paired_summary.columns]
    random_points = curves[curves.strategy == "random"][["method", "repetition", "labeled_count", *METRICS]]
    fixed = curves.merge(random_points.rename(columns={m: f"random_{m}" for m in METRICS}),
                         on=["method", "repetition", "labeled_count"], how="left")
    for metric in METRICS:
        fixed[f"difference_{metric}"] = fixed[metric] - fixed[f"random_{metric}"]
    means = curves.groupby(["method", "strategy", "labeled_count"])[list(METRICS)].agg(["mean", "std", "count"])
    means.columns = ["_".join(parts) for parts in means.columns]
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("learning_curves", curves), ("acquisition_events", event_frame),
                        ("full_pool_references", reference_frame), ("curve_areas", area_frame),
                        ("paired_comparisons", paired), ("fixed_budget_comparisons", fixed),
                        ("paired_summary", paired_summary.reset_index()),
                        ("mean_curves", means.reset_index())):
        frame.to_csv(output / f"{name}.csv", index=False)
    _plots(output, curves, event_frame, reference_frame, split, bool(design["smoke"]))
    save_json(output / "summary.json", {"status": "complete", "study_sha256": sha256(root / "study.json"),
              "split": split, "smoke": design["smoke"], "curves": len(areas), "events": len(events),
              "label_interval": [schedule[0], schedule[-1]],
              "uncertainty_band": "sample standard deviation across repetitions, omitted when n=1",
              "auc": "normalized trapezoidal valid mean disc/cup Dice; full-pool reference excluded",
              "pairing": "same repetition and method versus random; missing random arms produce missing comparisons",
              "time_axis": "elapsed wall time within each arm, including interruption downtime; not pooled across repetitions",
              "target_efficiency": "no target was predeclared; reporting fixed-budget comparisons and curve area",
              "files": {p.name: sha256(p) for p in output.glob("*.csv")}})
    return output


def _plots(output, curves, events, references, split, smoke):
    figures = output / "figures"
    figures.mkdir()
    colors = {"random": "#64748b", "bald": "#2563eb", "vcdr": "#d97706"}
    labels = {"valid_mean_dice": "Mean disc/cup Dice", "valid_dice_disc": "Disc Dice",
              "valid_dice_cup": "Cup Dice", "valid_vcdr_absolute_error": "vCDR MAE (valid cases)",
              "vcdr_coverage": "vCDR valid-case coverage"}
    methods = list(curves.method.unique())
    caption = ("SMOKE TEST — functional check only | " if smoke else "") + split.upper()
    for metric, ylabel in labels.items():
        fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 4.5), squeeze=False, layout="constrained")
        for ax, method in zip(axes[0], methods):
            part = curves[curves.method == method]
            for strategy, frame in part.groupby("strategy"):
                grouped = frame.groupby("labeled_count")[metric].agg(["mean", "std", "count"])
                x = grouped.index.to_numpy()
                mean, std = grouped["mean"].to_numpy(), grouped["std"].to_numpy()
                ax.plot(x, mean, "o-", color=colors[strategy], label=strategy.upper())
                if (grouped["count"] > 1).any():
                    ax.fill_between(x, mean - std, mean + std, color=colors[strategy], alpha=.12)
                requested = frame[frame.requests_more_images].labeled_count.unique()
                for n in requested:
                    ax.plot(n, grouped.loc[n, "mean"], marker="s", markersize=8,
                            markerfacecolor="none", markeredgecolor=colors[strategy])
            for n in sorted(part[part.requests_more_images].labeled_count.unique()):
                requested = events[(events.method == method) & (events.labeled_before == n)]
                batch = int(requested.selected_count.iloc[0])
                ax.axvline(n, color="#94a3b8", linestyle=":", linewidth=.7)
                ax.text(n, .98, f"+{batch}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            reference = references[references.method == method][metric].mean()
            if np.isfinite(reference):
                ax.axhline(reference, color="#475569", linestyle="--", label="Full-pool reference")
            ax.set(xlabel="Labeled training images (squares mark acquisition)", ylabel=ylabel, title=method.replace("_", " "))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(caption + " — " + ylabel, fontsize=11)
        fig.savefig(figures / f"{metric}.png", dpi=150)
        fig.savefig(figures / f"{metric}.pdf")
        plt.close(fig)
    for (method, strategy, repetition), part in curves.groupby(["method", "strategy", "repetition"]):
        part = part.sort_values("round")
        subset = events[(events.method == method) & (events.strategy == strategy) & (events.repetition == repetition)]
        stem = f"{method}_{strategy}_repeat_{repetition:02d}"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, metric in zip(axes, ("valid_mean_dice", "valid_vcdr_absolute_error")):
            ax.plot(part.elapsed_wall_seconds / 60, part[metric], "o-", color=colors[strategy])
            for event in subset.itertuples():
                ax.axvline(event.elapsed_wall_seconds / 60, color="#94a3b8", linestyle=":")
                ax.text(event.elapsed_wall_seconds / 60, .98, f"R{event.source_round}: +{event.selected_count}",
                        transform=ax.get_xaxis_transform(), rotation=90, va="top", fontsize=8)
            ax.set(xlabel="Elapsed wall time (minutes; includes downtime)", ylabel=labels[metric])
            ax.grid(alpha=.2)
        fig.suptitle(f"{caption} | {method} / {strategy} / repetition {repetition}", fontsize=10)
        fig.savefig(figures / f"{stem}_time.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        bottom = np.zeros(len(subset))
        for column in sorted(c for c in subset.columns if c.startswith("selected_") and c not in ("selected_count", "selected_ids")):
            counts = subset[column].fillna(0).to_numpy()
            axes[0].bar(subset.source_round, counts, bottom=bottom, label=column.removeprefix("selected_"))
            bottom += counts
        axes[0].set(xlabel="Acquisition round", ylabel="Selected images")
        axes[0].legend()
        axes[1].plot(subset.source_round, subset.invalid_score_fraction, "o-", label="Invalid score fraction")
        axes[1].plot(subset.source_round, subset.fallback_count / subset.selected_count, "s--", label="Fallback fraction")
        axes[1].set(xlabel="Acquisition round", ylabel="Fraction", ylim=(-.03, 1.03))
        axes[1].legend()
        fig.suptitle(f"{caption} | {method} / {strategy} / repetition {repetition}", fontsize=10)
        fig.savefig(figures / f"{stem}_acquisitions.png", dpi=150)
        plt.close(fig)

"""Active-learning stages, recoverable studies, frozen test evaluation and graphs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import torch

from dp2.common.uncertainty import SCORING_PROTOCOL
from .acquisition import select_images
from .annotation import reveal_selection
from .protocol import Experiment, prepare_experiment
from .scoring import export_pool_scores
from .state import load_state, read_csv, save_json, seed_for, sha256
from .training import load_predictor, state_provenance, train_subset

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Audit and freeze data; reveal initial subsets")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/active_learning.json")
    prepare.add_argument("--paths", type=Path, default=ROOT / "configs/paths.json")
    prepare.add_argument("--data-root", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--smoke", action="store_true", help="8 initial labels, 2x2 acquisitions, one repetition; smoke-only")
    run = commands.add_parser("run", help="Run/resume the declared experiment grid and full-pool references")
    run.add_argument("--experiment", type=Path, required=True)
    run.add_argument("--data-root", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--methods", nargs="+", choices=("mc_dropout", "deep_ensembles"))
    run.add_argument("--strategies", nargs="+", choices=("random", "bald", "vcdr"))
    run.add_argument("--repetitions", nargs="+", type=int)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--sequential", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--resume", action="store_true")
    for name in ("freeze", "evaluate-study", "summarize"):
        command = commands.add_parser(name)
        command.add_argument("--experiment", type=Path, required=True)
        command.add_argument("--data-root", type=Path)
        command.add_argument("--study", type=Path, required=True)
        if name == "evaluate-study":
            command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            command.add_argument("--sequential", action="store_true")
        if name == "summarize":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--split", choices=("val", "test"), default="val")
            command.add_argument("--include-smoke", action="store_true")
    stages = {}
    for name in ("train", "score", "select", "reveal"):
        stage = commands.add_parser(name)
        stages[name] = stage
        stage.add_argument("--experiment", type=Path, required=True)
        stage.add_argument("--state", type=Path, required=True)
        stage.add_argument("--data-root", type=Path, help="Override local dataset root, retaining relative paths and IDs")
        stage.add_argument("--output", type=Path, required=True)
    stages["train"].add_argument("--method", choices=("mc_dropout", "deep_ensembles"), required=True)
    stages["train"].add_argument("--smoke", action="store_true", help="1 epoch, 128px, random initialization; uses all current labeled images")
    for name in ("train", "score"):
        stages[name].add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    stages["score"].add_argument("--predictor", type=Path, required=True)
    stages["score"].add_argument("--sequential", action="store_true", help="Hold ensemble members on CPU, transfer one at a time")
    stages["select"].add_argument("--method", choices=("mc_dropout", "deep_ensembles"), required=True)
    stages["select"].add_argument("--strategy", choices=("random", "bald", "vcdr"), required=True)
    stages["select"].add_argument("--scores", type=Path, help="Completed score directory; omitted for random")
    stages["select"].add_argument("--count", type=int, help="Default: frozen acquisition size")
    stages["reveal"].add_argument("--selection", type=Path, required=True, help="Completed selection directory")
    args = parser.parse_args(argv)
    if args.command in ("run", "freeze", "evaluate-study", "summarize"):
        from .runner import run_study, freeze_study, evaluate_frozen_study
        experiment = Experiment(args.experiment, args.data_root)
        if args.command == "run":
            result = run_study(experiment, args.output, args.methods, args.strategies, args.repetitions,
                               args.device, args.sequential, args.smoke, args.resume)
        elif args.command == "freeze":
            result = freeze_study(experiment, args.study)
        elif args.command == "evaluate-study":
            result = evaluate_frozen_study(experiment, args.study, args.device, args.sequential)
        else:
            from .analysis import summarize_study
            result = summarize_study(experiment, args.study, args.output, args.split, args.include_smoke)
        print("Saved:", result, flush=True)
        return
    if args.command == "prepare":
        config = json.loads(args.config.read_text())
        if args.smoke:
            config.update(initial_size=8, acquisition_size=2, acquisitions=2, repetitions=1,
                          mc_passes=2, ensemble_members=2, smoke=True)
        prepare_experiment(args.output, json.loads(args.paths.read_text()), config, args.data_root)
    else:
        experiment = Experiment(args.experiment, args.data_root)
        state, labeled, pool, _ = load_state(experiment, args.state)
        expected = state_provenance(experiment, state)
        if args.command == "train":
            train_subset(experiment, args.state, args.method, args.output, args.device, args.smoke)
        elif args.command == "score":
            models, cfg, manifest = load_predictor(args.predictor, expected, args.device, args.sequential)
            if state.get("smoke", False) and not manifest["smoke"]:
                raise ValueError("A smoke-derived state cannot use a research predictor")
            export_pool_scores(args.output, pool, models, cfg, manifest, experiment.data_root,
                               experiment.config["mc_passes"], experiment.config["seed"],
                               state["repetition"], state["round"], args.sequential)
        elif args.command == "select":
            args.output.mkdir(parents=True, exist_ok=False)
            report = {"status": "running", "provenance": expected,
                      "method": args.method, "strategy": args.strategy,
                      "smoke": bool(state.get("smoke", False) or experiment.config.get("smoke", False)),
                      "requested_count": args.count if args.count is not None else experiment.config["acquisition_size"],
                      "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            save_json(args.output / "selection.json", report)
            try:
                scores = None
                if args.strategy == "random":
                    if args.scores is not None:
                        raise ValueError("Random selection must not receive scoring inputs")
                else:
                    if args.scores is None:
                        raise ValueError("Uncertainty selection requires --scores")
                    source = json.loads((args.scores / "report.json").read_text())
                    if (source.get("status") != "complete" or source["provenance"] != expected
                            or source["method"] != args.method
                            or source["pool_ids_sha256"] != state["pool_ids_sha256"]
                            or source["scoring_protocol"] != SCORING_PROTOCOL
                            or sha256(args.scores / "pool_scores.csv") != source["scores_sha256"]):
                        raise ValueError("Scores are incomplete, changed, or belong to a different pool/predictor")
                    scores = pd.read_csv(args.scores / "pool_scores.csv", dtype={"image_id": str})
                    report.update(source_scores_directory=str(args.scores.resolve()),
                                  source_report_sha256=sha256(args.scores / "report.json"),
                                  predictor_directory=source["predictor_directory"],
                                  members=source["members"], smoke=report["smoke"] or source["smoke"])
                # Excludes method: random arms get identical trajectories for paired repeats.
                acquisition_seed = seed_for(experiment.config["seed"], "acquisition",
                                            state["repetition"], state["round"], args.strategy)
                selected, ranking = select_images(pool.image_id, args.strategy,
                                                   report["requested_count"], acquisition_seed, scores)
                selected.to_csv(args.output / "acquisition.csv", index=False)
                ranking.to_csv(args.output / "ranking.csv", index=False)
                report.update(status="complete", acquisition_seed=acquisition_seed,
                              selected_count=len(selected), fallback_count=int(selected.fallback.sum()),
                              invalid_score_count=int((~ranking.score_valid).sum()) if scores is not None else None,
                              acquisition_sha256=sha256(args.output / "acquisition.csv"),
                              ranking_sha256=sha256(args.output / "ranking.csv"))
            except Exception as error:
                report.update(status="failed", error=str(error))
                save_json(args.output / "selection.json", report)
                raise
            save_json(args.output / "selection.json", report)
        else:
            report = json.loads((args.selection / "selection.json").read_text())
            if (report.get("status") != "complete" or report["provenance"] != expected
                    or sha256(args.selection / "acquisition.csv") != report["acquisition_sha256"]):
                raise ValueError("Selection is incomplete, modified, or belongs to another state")
            selected = read_csv(args.selection / "acquisition.csv")
            if len(selected) != report["selected_count"]:
                raise ValueError("Selection row count mismatch")
            reveal_selection(experiment, args.state, selected.image_id, args.output,
                             smoke=report.get("smoke", False),
                             acquisition={"selection_sha256": sha256(args.selection / "selection.json"),
                                          "parent_state_sha256": sha256(args.state / "state.json"),
                                          "method": report["method"], "strategy": report["strategy"]})
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()

"""Round recovery, method evaluation, test isolation and graph-data joins."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from test_active_learning import make_fixture, tiny_model
from dp2.active_learning.analysis import curve_area, summarize_study
from dp2.active_learning.artifacts import stage, verify_stage, rename_directory
from dp2.active_learning.evaluation import image_metrics, aggregate_metrics, evaluate_predictor
from dp2.active_learning.protocol import prepare_experiment
from dp2.active_learning.runner import run_study, freeze_study, evaluate_frozen_study
from dp2.active_learning.state import read_csv, sha256
from dp2.active_learning.training import train_subset


class EvaluationTests(unittest.TestCase):
    def test_transient_windows_directory_lock_is_retried(self):
        error = PermissionError("temporary directory lock")
        error.winerror = 32
        with patch("dp2.active_learning.artifacts.os.name", "nt"), patch.object(Path, "rename", side_effect=[error, Path("published")]) as rename, patch("dp2.active_learning.artifacts.time.sleep") as sleep:
            self.assertEqual(rename_directory(Path("pending"), Path("published")), Path("published"))
        self.assertEqual(rename.call_count, 2)
        sleep.assert_called_once_with(.05)

    def test_padding_geometry_and_missing_disc_coverage(self):
        reference = np.zeros((8, 8), dtype=np.int64)
        reference[2:7, 2:7] = 1
        reference[3:5, 3:5] = 2
        pred = reference.copy()
        pred[0, :] = 2
        valid = np.ones_like(pred, dtype=bool)
        valid[0, :] = False
        metrics = image_metrics(pred, reference, valid)
        self.assertEqual(metrics["valid_mean_dice"], 1.)
        self.assertLess(metrics["legacy_mean_dice"], 1.)
        self.assertEqual(metrics["valid_vcdr_absolute_error"], 0.)
        failed = image_metrics(np.zeros_like(pred), reference, valid)
        summary = aggregate_metrics(pd.DataFrame([metrics, failed]))
        self.assertEqual(summary["missing_disc_count"], 1)
        self.assertEqual(summary["vcdr_coverage"], .5)
        self.assertEqual(summary["valid_vcdr_absolute_error"], 0.)
        empty = aggregate_metrics(pd.DataFrame([failed]))
        self.assertIsNone(empty["valid_vcdr_absolute_error"])
        self.assertEqual(empty["vcdr_valid_count"], 0)

    def test_normalized_area_and_invalid_curves(self):
        self.assertAlmostEqual(curve_area([150, 250, 350], [.2, .4, .6]), .4)
        for labels, values in (([150, 150], [.2, .3]), ([1, 2], [.2, np.nan]), ([1], [.2])):
            with self.assertRaises(ValueError):
                curve_area(labels, values)


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        paths, config = make_fixture(self.root / "data")
        config["smoke"] = True
        self.experiment = prepare_experiment(self.root / "experiment", paths, config)
        self.initial = self.experiment.directory / "repeat_00/initial"
        for name in ("dp2.common.training.build_model", "dp2.active_learning.training.build_model"):
            mock = patch(name, side_effect=tiny_model)
            mock.start()
            self.addCleanup(mock.stop)

    def test_completed_member_is_reused_after_interruption(self):
        from dp2.active_learning.training import train_model as real_train
        calls = []

        def interrupted(*args, **kwargs):
            calls.append(args[3])
            if len(calls) == 2:
                raise RuntimeError("member interruption")
            return real_train(*args, **kwargs)

        output = self.root / "ensemble"
        with patch("dp2.active_learning.training.train_model", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                train_subset(self.experiment, self.initial, "deep_ensembles", output)
            original = json.loads((output / "ensemble.json").read_text())["members"][0]
            train_subset(self.experiment, self.initial, "deep_ensembles", output, resume=True)
        final = json.loads((output / "ensemble.json").read_text())
        self.assertEqual(final["members"][0], original)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1], calls[2])

    def test_selection_and_commit_recovery_preserve_event_and_membership(self):
        root = self.root / "study"

        def stop_after_selection(name, index):
            if name == "selected" and index == 0:
                raise RuntimeError("selection interruption")

        with self.assertRaisesRegex(RuntimeError, "selection interruption"):
            run_study(self.experiment, root, ["mc_dropout"], ["random"], [0], hook=stop_after_selection)
        event = root / "repeat_00/mc_dropout/random/round_00/event/acquisition_checkpoint.json"
        digest = sha256(event)
        actual_rename = Path.rename
        failed = []

        def interrupted_rename(path, target):
            if path.name == "state.partial" and path.parent.name == "round_01" and not failed:
                failed.append(True)
                raise RuntimeError("commit interruption")
            return actual_rename(path, target)

        with patch.object(Path, "rename", interrupted_rename):
            with self.assertRaisesRegex(RuntimeError, "commit interruption"):
                run_study(self.experiment, root, ["mc_dropout"], ["random"], [0], resume=True)
        self.assertTrue((event.parents[2] / "round_01/state.partial/_complete.json").exists())
        run_study(self.experiment, root, ["mc_dropout"], ["random"], [0], resume=True)
        self.assertEqual(sha256(event), digest)
        events = list(root.glob("repeat_*/**/acquisition_checkpoint.json"))
        self.assertEqual(len(events), 2)
        final_state = read_csv(root / "repeat_00/mc_dropout/random/round_02/state/labeled.csv")
        self.assertEqual(len(final_state), 8)
        self.assertFalse(final_state.image_id.duplicated().any())
        with patch("dp2.active_learning.runner.train_subset", side_effect=AssertionError("retrained")):
            run_study(self.experiment, root, ["mc_dropout"], ["random"], [0], resume=True)

    def test_all_methods_strategies_test_gate_and_graph_join(self):
        root = self.root / "study"
        original = self.experiment.private_annotations
        test_reads = []

        def watch(split):
            if split == "test":
                test_reads.append(split)
            return original(split)

        with patch.object(self.experiment, "private_annotations", side_effect=watch):
            run_study(self.experiment, root, repetitions=[0])
            self.assertEqual(test_reads, [])
            with self.assertRaises(FileNotFoundError):
                evaluate_frozen_study(self.experiment, root)
            self.assertEqual(test_reads, [])
            with self.assertRaises(ValueError):
                evaluate_predictor(self.experiment, self.initial, root, self.root / "forbidden", split="test")
            freeze_study(self.experiment, root)
            evaluate_frozen_study(self.experiment, root)
            self.assertTrue(test_reads)
        with self.assertRaises(ValueError):
            summarize_study(self.experiment, root, self.root / "not_research")
        with patch("dp2.active_learning.runner.train_subset", side_effect=AssertionError("summary retraining")), patch("dp2.active_learning.runner.score_stage", side_effect=AssertionError("summary scoring")):
            summarize_study(self.experiment, root, self.root / "summary", "test", include_smoke=True)
        curves = pd.read_csv(self.root / "summary/learning_curves.csv")
        events = pd.read_csv(self.root / "summary/acquisition_events.csv")
        self.assertEqual(len(curves), 18)
        self.assertEqual(len(events), 12)
        self.assertFalse(events.event_id.duplicated().any())
        self.assertEqual(set(events.labeled_before), {4, 6})
        self.assertEqual(set(events.labeled_after), {6, 8})
        self.assertTrue(events.after_valid_mean_dice.notna().all())
        for event in events.itertuples():
            point = curves[(curves.method == event.method) & (curves.strategy == event.strategy) & (curves.repetition == event.repetition) & (curves["round"] == event.source_round)].iloc[0]
            self.assertEqual(point.elapsed_wall_seconds, event.elapsed_wall_seconds)
        paired = pd.read_csv(self.root / "summary/paired_comparisons.csv")
        self.assertTrue((paired[paired.strategy == "random"].difference_from_random == 0).all())
        self.assertEqual(len(list((self.root / "summary/figures").glob("*.png"))), 17)
        shared_paths = curves[curves["round"] == 0].groupby("method").predictor.nunique()
        self.assertTrue((shared_paths == 1).all())
        for index in (0, 1):
            selected = []
            for method in ("mc_dropout", "deep_ensembles"):
                selected.append(read_csv(root / f"repeat_00/{method}/random/round_{index:02d}/selection/acquisition.csv").image_id.tolist())
            self.assertEqual(*selected)
        first_event = root / events.iloc[0].checkpoint_record
        first_event.write_text(first_event.read_text() + " ")
        with self.assertRaises(ValueError):
            evaluate_frozen_study(self.experiment, root)

    def test_completed_stage_tampering_is_rejected(self):
        folder = self.root / "stage"

        def build(path):
            path.mkdir()
            (path / "result.txt").write_text("original")

        stage(folder, build)
        (folder / "result.txt").write_text("modified")
        with self.assertRaises(ValueError):
            stage(folder, build)


if __name__ == "__main__":
    unittest.main()

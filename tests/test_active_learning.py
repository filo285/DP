"""Leakage boundaries, reproducible acquisitions and subset-training integration."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from dp2.baseline import Config, load_mask
from dp2.active_learning.acquisition import AllScoresInvalid, select_images
from dp2.active_learning.annotation import reveal_selection
from dp2.active_learning.cli import main
from dp2.active_learning.data import ImagePoolDataset
from dp2.active_learning.protocol import Experiment, prepare_experiment, sample_initial
from dp2.active_learning.scoring import score_pool
from dp2.active_learning.state import (ids_hash, load_state, read_csv, resolve_path,
                                       save_json, seed_for, sha256, validate_public)
from dp2.active_learning.training import load_predictor, state_provenance, train_subset
from dp2.deep_ensembles.inference import ensemble_predict


def make_fixture(root):
    """Eight train, two val and two test images with independent synthetic pixels."""
    root = Path(root)
    root.mkdir(parents=True)
    tables = {"origa": [], "refuge": []}
    for index in range(12):
        dataset = "origa" if index < 4 else "refuge"
        split = "train" if index < 8 else ("val" if index < 10 else "test")
        rng = np.random.default_rng(index)
        image = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[3:22, 6:27] = 1
        mask[8:16, 12:21] = 2
        if dataset == "refuge":
            mask = np.array([255, 128, 0], dtype=np.uint8)[mask]
        image_path, mask_path = root / f"image{index}.png", root / f"mask{index}.png"
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)
        tables[dataset].append({"image_id": f"{index:03d}", "image_path": str(image_path),
                                "mask_path": str(mask_path), "split": split,
                                "glaucoma": "hidden", "vertical_cdr_from_mask": "hidden"})
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root / f"{name}.csv", index=False)
    paths = {"dp1_root": str(root), "origa_table": "origa.csv", "refuge_table": "refuge.csv"}
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/active_learning.json").read_text())
    config.update(initial_size=4, acquisition_size=2, acquisitions=2, repetitions=2,
                  mc_passes=2, ensemble_members=2)
    config["training"].update(img_size=128, epochs=1, batch_size=2, encoder_weights=None)
    return paths, config


def tiny_model(cfg):
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Dropout2d(cfg.dropout), nn.Conv2d(4, 3, 1))
    with torch.no_grad():
        model[-1].bias.copy_(torch.tensor([-4., 3., 1.]))
    return model


class AcquisitionTests(unittest.TestCase):
    def test_separate_scores_ties_invalids_and_no_leakage(self):
        scores = pd.DataFrame({"image_id": ["a", "b", "c", "d"],
                               "bald": [.5, .2, np.nan, .2],
                               "image_uncertainty_valid": [True, True, False, True],
                               "vcdr_uncertainty": [.01, .9, np.nan, .1],
                               "vcdr_uncertainty_valid": [True, True, False, True]})
        bald, _ = select_images(scores.image_id, "bald", 1, 8, scores)
        vcdr, _ = select_images(scores.image_id, "vcdr", 1, 8, scores)
        self.assertEqual(bald.image_id.tolist(), ["a"])
        self.assertEqual(vcdr.image_id.tolist(), ["b"])
        first, _ = select_images(scores.image_id, "bald", 4, 8, scores)
        second, _ = select_images(list(reversed(scores.image_id)), "bald", 4, 8, scores.iloc[::-1])
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.iloc[-1].image_id, "c")
        self.assertTrue(first.iloc[-1].fallback)
        self.assertTrue(np.isnan(first.iloc[-1].score))
        for name in ("mask_path", "dice_cup", "glaucoma", "vcdr_reference"):
            with self.assertRaises(ValueError):
                select_images(scores.image_id, "bald", 2, 8, scores.assign(**{name: 1}))
        invalid = scores.copy()
        invalid["image_uncertainty_valid"] = False
        with self.assertRaises(AllScoresInvalid):
            select_images(scores.image_id, "bald", 2, 8, invalid)
        with self.assertRaises(ValueError):
            select_images(["a", "b"], "bald", 2, 8, scores)

    def test_random_replay_and_pool_exhaustion(self):
        a, _ = select_images(["a", "b", "c"], "random", 9, 41)
        b, _ = select_images(["c", "a", "b"], "random", 9, 41)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.image_id), {"a", "b", "c"})
        self.assertFalse(a.fallback.any())
        with self.assertRaises(ValueError):
            select_images([], "random", 2, 0)
        with self.assertRaises(ValueError):
            select_images(["a", "a"], "random", 2, 0)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths, self.config = make_fixture(self.root / "data")
        self.experiment = prepare_experiment(self.root / "experiment", self.paths, self.config)
        self.initial = self.experiment.directory / "repeat_00/initial"

    def tearDown(self):
        self.tmp.cleanup()

    def test_frozen_splits_initial_reveal_and_portability(self):
        state, labeled, pool, annotations = load_state(self.experiment, self.initial)
        self.assertEqual([len(self.experiment.splits[k]) for k in ("train", "val", "test")], [8, 2, 2])
        self.assertEqual(labeled.groupby("dataset").size().to_dict(), {"origa": 2, "refuge": 2})
        self.assertEqual(set(labeled.image_id), set(annotations.image_id))
        self.assertFalse(set(labeled.image_id) & set(pool.image_id))
        self.assertEqual(state["labeled_ids_sha256"], ids_hash(labeled.image_id))
        with self.assertRaises(ValueError):
            self.experiment.annotations.reveal(self.experiment.splits["test"].image_id)
        with self.assertRaises(ValueError):
            validate_public(pool.assign(mask_path="forbidden"))
        moved = self.root / "moved_data"
        shutil.copytree(self.root / "data", moved)
        dataset = ImagePoolDataset(pool, Config(img_size=32), moved)
        self.assertEqual(dataset[0][1].shape, (3, 32, 32))
        with self.assertRaises(ValueError):
            resolve_path(moved, "../outside.png")
        altered = read_csv(self.initial / "labeled.csv")
        altered.loc[0, "image_id"] = "bad"
        altered.to_csv(self.initial / "labeled.csv", index=False)
        with self.assertRaises(ValueError):
            load_state(self.experiment, self.initial)

    def test_duplicate_content_and_patient_overlap_rejected(self):
        shutil.copyfile(self.root / "data/image0.png", self.root / "data/image8.png")
        with self.assertRaisesRegex(ValueError, "audit failed"):
            prepare_experiment(self.root / "bad_content", self.paths, self.config)
        audit = json.loads((self.root / "bad_content/audit.json").read_text())
        self.assertIn("pixel_sha256", audit["failures"])
        table = read_csv(self.root / "data/refuge.csv")
        table["patient_id"] = "same_patient"
        table.to_csv(self.root / "data/refuge.csv", index=False)
        with self.assertRaisesRegex(ValueError, "audit failed"):
            prepare_experiment(self.root / "bad_patient", self.paths, self.config)
        audit = json.loads((self.root / "bad_patient/audit.json").read_text())
        self.assertIn("patient_overlap", audit["failures"])

    def test_mask_free_scoring_stable_image_seeds_and_padding(self):
        _, _, pool, _ = load_state(self.experiment, self.initial)
        for path in (self.root / "data").glob("mask*.png"):
            path.unlink()  # No masks exist while the pool is scored.
        cfg = Config(img_size=32, device="cpu")
        model = tiny_model(cfg)
        dataset = ImagePoolDataset(pool, cfg, self.root / "data")
        self.assertFalse(dataset[0][2].all())
        with patch("dp2.baseline.load_mask", side_effect=AssertionError("Pool tried to read a mask")):
            first = score_pool(pool, [model], cfg, "mc_dropout", self.root / "data", passes=3)
            second = score_pool(pool.iloc[::-1], [model], cfg, "mc_dropout", self.root / "data", passes=3)
        pd.testing.assert_frame_equal(first.sort_values("image_id").reset_index(drop=True),
                                      second.sort_values("image_id").reset_index(drop=True))
        self.assertNotIn("mask_path", first)
        self.assertNotIn("dice_disc", first)

    def test_reveal_membership_and_duplicate_rejection(self):
        _, labeled, pool, _ = load_state(self.experiment, self.initial)
        selected = pool.image_id.iloc[:2].tolist()
        reveal_selection(self.experiment, self.initial, selected, self.root / "next", smoke=True)
        next_state, next_labeled, next_pool, annotations = load_state(self.experiment, self.root / "next")
        self.assertTrue(next_state["smoke"])
        self.assertEqual(len(next_labeled), len(labeled) + 2)
        self.assertEqual(len(next_pool), len(pool) - 2)
        self.assertEqual(set(annotations.image_id), set(next_labeled.image_id))
        reveal_selection(self.experiment, self.root / "next", next_pool.image_id, self.root / "final")
        final_state, _, final_pool, _ = load_state(self.experiment, self.root / "final")
        self.assertTrue(final_state["smoke"])
        self.assertTrue(final_pool.empty)
        for ids in (selected * 2, labeled.image_id.iloc[:1].tolist(), self.experiment.splits["val"].image_id.tolist()):
            with self.assertRaises(ValueError):
                reveal_selection(self.experiment, self.initial, ids, self.root / "invalid")

    def test_subset_training_provenance_and_sequential_scoring(self):
        state, labeled, pool, annotations = load_state(self.experiment, self.initial)
        seen_masks = []

        def traced(path):
            seen_masks.append(Path(path).name)
            return load_mask(path)

        with patch("dp2.common.training.build_model", side_effect=tiny_model), patch("dp2.baseline.load_mask", side_effect=traced):
            for method in ("mc_dropout", "deep_ensembles"):
                train_subset(self.experiment, self.initial, method, self.root / method)
        allowed = {Path(p).name for p in annotations.mask_path}
        allowed.update(Path(p).name for p in self.experiment.private_annotations("val").mask_path)
        self.assertEqual(set(seen_masks), allowed)
        expected = state_provenance(self.experiment, state)
        with patch("dp2.active_learning.training.build_model", side_effect=tiny_model):
            for method in ("mc_dropout", "deep_ensembles"):
                models, cfg, manifest = load_predictor(self.root / method, expected, sequential=True)
                self.assertEqual(len(models), 1 if method == "mc_dropout" else 2)
                scores = score_pool(pool, models, cfg, method, self.root / "data", passes=2, sequential=True)
                self.assertEqual(len(scores), len(pool))
                with self.assertRaises(ValueError):
                    load_predictor(self.root / method, {**expected, "labeled_ids_sha256": "wrong"})
            image = torch.randn(1, 3, 16, 16)
            regular = ensemble_predict(models, image)
            sequential = ensemble_predict(models, image, sequential=True)
            torch.testing.assert_close(regular.mean_probabilities, sequential.mean_probabilities)
            manifest_path = self.root / "mc_dropout/model.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["scope"] = "full_pool_supervised"
            save_json(manifest_path, manifest)
            with self.assertRaises(ValueError):
                load_predictor(self.root / "mc_dropout", expected)
            ensemble_manifest = json.loads((self.root / "deep_ensembles/ensemble.json").read_text())
            member_path = self.root / "deep_ensembles" / ensemble_manifest["members"][0]["checkpoint"]
            with member_path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                load_predictor(self.root / "deep_ensembles", expected)

    def test_cli_scoring_selection_reveal_and_stale_score_rejection(self):
        base = ["--experiment", str(self.experiment.directory), "--state", str(self.initial)]
        with patch("dp2.common.training.build_model", side_effect=tiny_model), patch("dp2.active_learning.training.build_model", side_effect=tiny_model):
            main(["train", *base, "--method", "mc_dropout", "--device", "cpu", "--smoke", "--output", str(self.root / "model")])
            main(["score", *base, "--predictor", str(self.root / "model"), "--device", "cpu", "--output", str(self.root / "scores")])
        for strategy in ("bald", "vcdr", "random"):
            args = ["select", *base, "--method", "mc_dropout", "--strategy", strategy, "--output", str(self.root / strategy)]
            if strategy != "random":
                args += ["--scores", str(self.root / "scores")]
            main(args)
        main(["reveal", *base, "--selection", str(self.root / "bald"), "--output", str(self.root / "next")])
        next_state = load_state(self.experiment, self.root / "next")[0]
        self.assertEqual(next_state["labeled_count"], 6)
        self.assertTrue(next_state["smoke"])
        self.assertEqual(next_state["acquisition"]["strategy"], "bald")
        with self.assertRaises(ValueError):
            main(["select", "--experiment", str(self.experiment.directory), "--state", str(self.root / "next"),
                  "--method", "mc_dropout", "--strategy", "bald", "--scores", str(self.root / "scores"),
                  "--output", str(self.root / "stale")])
        self.assertEqual(json.loads((self.root / "stale/selection.json").read_text())["status"], "failed")
        scores_path = self.root / "scores/pool_scores.csv"
        invalid = pd.read_csv(scores_path)
        invalid["image_uncertainty_valid"] = False
        invalid.to_csv(scores_path, index=False)
        report = json.loads((self.root / "scores/report.json").read_text())
        report["scores_sha256"] = sha256(scores_path)
        save_json(self.root / "scores/report.json", report)
        with self.assertRaises(AllScoresInvalid):
            main(["select", *base, "--method", "mc_dropout", "--strategy", "bald",
                  "--scores", str(self.root / "scores"), "--output", str(self.root / "all_invalid")])
        self.assertEqual(json.loads((self.root / "all_invalid/selection.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

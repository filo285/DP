"""Ensemble numerics, inference isolation and checkpoint provenance."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from torch import nn
from dp2.common.uncertainty import entropy
from dp2.deep_ensembles.inference import ensemble_predict
from dp2.deep_ensembles.checkpoints import load_ensemble, sha256
from dp2.deep_ensembles.train import segmentation_loss
from dp2.baseline import Config


class EnsembleTests(unittest.TestCase):
    def test_reference_determinism_and_restoration(self):
        torch.manual_seed(7)
        models = [nn.Sequential(nn.BatchNorm2d(3), nn.Dropout2d(.8), nn.Conv2d(3, 3, 1)) for _ in range(3)]
        images = torch.randn(2, 3, 8, 8)
        for model in models:
            model.eval()
        with torch.inference_mode():
            probabilities = torch.stack([model(images).softmax(1) for model in models])
        for model in models:
            model.train()
            model[2].eval()
        modes = [[m.training for m in model.modules()] for model in models]
        buffers = [[v.clone() for v in model.buffers()] for model in models]
        rng = torch.get_rng_state().clone()
        result = ensemble_predict(models, images)
        torch.testing.assert_close(result.mean_probabilities, probabilities.mean(0))
        torch.testing.assert_close(result.std_probabilities, probabilities.std(0, correction=0))
        expected = torch.stack([entropy(p) for p in probabilities]).mean(0)
        torch.testing.assert_close(result.mutual_information, (entropy(probabilities.mean(0)) - expected).clamp_min(0), atol=3e-7, rtol=1e-4)
        torch.testing.assert_close(result.mean_probabilities, ensemble_predict(models, images).mean_probabilities, atol=0, rtol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for index, model in enumerate(models):
            self.assertEqual(modes[index], [m.training for m in model.modules()])
            for before, after in zip(buffers[index], model.buffers()):
                self.assertTrue(torch.equal(before, after))

    def test_identical_predictions_zero_disagreement(self):
        models = [nn.Conv2d(3, 3, 1) for _ in range(2)]
        models[1].load_state_dict(models[0].state_dict())
        result = ensemble_predict(models, torch.ones(1, 3, 8, 8))
        self.assertEqual(result.std_probabilities.count_nonzero().item(), 0)
        self.assertEqual(result.mutual_information.count_nonzero().item(), 0)

    def test_bad_members_and_exception_restoration(self):
        model = nn.Conv2d(3, 3, 1)
        images = torch.ones(1, 3, 8, 8)
        for models in ([], [model], [model, model]):
            with self.assertRaises(ValueError):
                ensemble_predict(models, images)
        bad = nn.Conv2d(3, 4, 1).train()
        with self.assertRaises(ValueError):
            ensemble_predict([model, bad], images)
        self.assertTrue(model.training)
        self.assertTrue(bad.training)
        with torch.no_grad():
            bad.weight.fill_(float("nan"))
        with self.assertRaises(ValueError):
            ensemble_predict([model, bad], images)
        self.assertTrue(bad.training)

    def test_loss_supports_training(self):
        model = nn.Conv2d(3, 3, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters())
        loss = segmentation_loss(model(torch.randn(2, 3, 8, 8)), torch.randint(0, 3, (2, 8, 8)))
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.weight))

    def test_checkpoint_integrity_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"status": "complete", "split_sha256": {"val.csv": "test"}, "members": []}
            for seed in (1, 2):
                torch.manual_seed(seed)
                checkpoint = root / f"{seed}.pth"
                torch.save({"model": nn.Conv2d(3, 3, 1).state_dict(), "cfg": asdict(Config()),
                            "seed": seed, "split_sha256": manifest["split_sha256"]}, checkpoint)
                manifest["members"].append({"seed": seed, "checkpoint": checkpoint.name, "sha256": sha256(checkpoint)})
            path = root / "ensemble.json"
            path.write_text(json.dumps(manifest))
            with patch("dp2.deep_ensembles.checkpoints.build_model", side_effect=lambda cfg: nn.Conv2d(3, 3, 1)):
                models, cfg, _ = load_ensemble(root)
                self.assertEqual(len(models), 2)
                self.assertIsNone(cfg.encoder_weights)
                manifest["members"][1]["sha256"] = "corrupt"
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_ensemble(root)
                manifest["members"][1]["seed"] = 1
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    load_ensemble(root)


if __name__ == "__main__":
    unittest.main()

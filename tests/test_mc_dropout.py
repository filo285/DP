"""Numerical and state-safety checks; no data or checkpoints required."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch import nn
from dp2.common.uncertainty import entropy
from dp2.common.vcdr import vertical_cup_to_disc
from dp2.mc_dropout.inference import dropout_inference, mc_dropout_predict


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout2d(.4), nn.Conv2d(3, 3, 1))
        self.x = torch.randn(2, 3, 8, 8)

    def test_matches_stacked_reference_and_preserves_state(self):
        self.model.train()
        self.model[2].eval()  # mixed mode must also be restored
        modes = [m.training for m in self.model.modules()]
        buffers = {n: b.clone() for n, b in self.model.named_buffers()}
        rng = torch.get_rng_state().clone()
        result = mc_dropout_predict(self.model, self.x, passes=20, seed=5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        for n, b in self.model.named_buffers():
            self.assertTrue(torch.equal(buffers[n], b))
        with torch.random.fork_rng(), torch.inference_mode(), dropout_inference(self.model):
            torch.manual_seed(5)
            samples = torch.stack([self.model(self.x).softmax(1) for _ in range(20)])
        torch.testing.assert_close(result.mean_probabilities, samples.mean(0))
        torch.testing.assert_close(result.std_probabilities, samples.std(0, correction=0))
        expected_vcdr = torch.stack([vertical_cup_to_disc(p.argmax(1))[0] for p in samples])
        torch.testing.assert_close(result.vcdr_samples, expected_vcdr, equal_nan=True)
        expected = torch.stack([entropy(p) for p in samples]).mean(0)
        torch.testing.assert_close(result.expected_entropy, expected)
        torch.testing.assert_close(result.mutual_information,
                                   (entropy(samples.mean(0)) - expected).clamp_min(0), atol=3e-7, rtol=1e-4)
        self.assertGreater(result.std_probabilities.max().item(), 0)
        self.assertFalse(result.mean_probabilities.requires_grad)
        repeated = mc_dropout_predict(self.model, self.x, 20, seed=5)
        torch.testing.assert_close(result.mean_probabilities, repeated.mean_probabilities, rtol=0, atol=0)

    def test_exception_restores_modes(self):
        self.model.train()
        with self.assertRaises(RuntimeError):
            with dropout_inference(self.model):
                self.assertFalse(self.model[0].training)
                raise RuntimeError("simulated forward failure")
        self.assertTrue(all(m.training for m in self.model.modules()))

    def test_unseeded_calls_advance_rng(self):
        before = torch.get_rng_state().clone()
        mc_dropout_predict(self.model, self.x, 3)
        self.assertFalse(torch.equal(before, torch.get_rng_state()))

    def test_identical_samples_have_zero_disagreement(self):
        with torch.no_grad():
            self.model[2].weight.zero_()
            self.model[2].bias.copy_(torch.tensor([4., 0., 0.]))
        result = mc_dropout_predict(self.model, self.x, 3, seed=1)
        self.assertEqual(result.std_probabilities.count_nonzero().item(), 0)
        self.assertLess(result.mutual_information.max().item(), 1e-6)
        valid = torch.zeros(2, 8, 8, dtype=torch.bool)
        valid[:, 2:6, 2:6] = True
        scores = result.image_scores(valid)
        self.assertTrue(scores["empty_foreground"].all())
        self.assertTrue((scores["roi_pixels"] == 0).all())
        self.assertFalse(scores["image_uncertainty_valid"].any())
        for name in ("std", "entropy", "bald", "image_uncertainty"):
            self.assertTrue(torch.isnan(scores[name]).all())
        with self.assertRaises(ValueError):
            result.image_scores(torch.zeros_like(valid))

    def test_rejects_invalid_passes_and_absent_dropout(self):
        for passes in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                mc_dropout_predict(self.model, self.x, passes)
        with self.assertRaises(ValueError):
            mc_dropout_predict(nn.Conv2d(3, 3, 1), self.x)


if __name__ == "__main__":
    unittest.main()

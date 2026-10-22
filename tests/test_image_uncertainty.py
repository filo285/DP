"""Known-geometry checks for per-image BALD and vCDR uncertainty."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch import nn
from dp2.common.vcdr import vertical_cup_to_disc, summarize_vcdr
from dp2.common.uncertainty import ProbabilityAccumulator, UncertaintyResult
from dp2.deep_ensembles.inference import ensemble_predict


def segmentation(cup_height):
    mask = torch.zeros(1, 12, 8, dtype=torch.long)
    mask[:, 1:11, 1:7] = 1  # Full disc height 10, width 6.
    mask[:, 3:3 + cup_height, 2:6] = 2
    return mask


def logits_for(mask):
    return torch.nn.functional.one_hot(mask, 3).permute(0, 3, 1, 2).float() * 20


class FixedModel(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("logits", logits_for(mask))

    def forward(self, images):
        return self.logits.expand(len(images), -1, -1, -1)


class ImageUncertaintyTests(unittest.TestCase):
    def test_vertical_not_area_ratio_and_full_disc(self):
        mask = segmentation(4)
        ratio, disc, cup = vertical_cup_to_disc(mask)
        self.assertEqual(disc.item(), 10)
        self.assertEqual(cup.item(), 4)
        self.assertAlmostEqual(ratio.item(), .4)
        # Cup width changes without changing the vertical ratio.
        mask[mask == 2] = 1
        mask[:, 3:7, 3] = 2
        self.assertAlmostEqual(vertical_cup_to_disc(mask)[0].item(), .4)

    def test_padding_and_empty_structures(self):
        mask = segmentation(4)
        mask[:, 0] = 2
        valid = torch.ones_like(mask, dtype=torch.bool)
        valid[:, 0] = False
        self.assertAlmostEqual(vertical_cup_to_disc(mask, valid)[0].item(), .4)
        batch = torch.cat([torch.zeros_like(mask), segmentation(0), segmentation(4)])
        ratios, _, _ = vertical_cup_to_disc(batch)
        self.assertTrue(torch.isnan(ratios[0]))
        self.assertEqual(ratios[1].item(), 0)
        self.assertAlmostEqual(ratios[2].item(), .4)
        with self.assertRaises(ValueError):
            vertical_cup_to_disc(mask, torch.zeros_like(valid))

    def test_population_std_and_invalid_policy(self):
        samples = torch.tensor([[.4, .5, float("nan")], [.6, float("nan"), float("nan")]])
        summary = summarize_vcdr(samples)
        self.assertAlmostEqual(summary["vcdr_mean"][0].item(), .5)
        self.assertAlmostEqual(summary["vcdr_uncertainty"][0].item(), .1)
        self.assertTrue(torch.isnan(summary["vcdr_uncertainty"][1:]).all())
        self.assertEqual(summary["vcdr_invalid_fraction"].tolist(), [0., .5, 1.])
        self.assertTrue(torch.isnan(summarize_vcdr(samples[:1])["vcdr_uncertainty"]).all())
        self.assertEqual(summarize_vcdr(torch.tensor([[.7], [.7]]))["vcdr_uncertainty"].item(), 0.)

    def test_accumulator_records_each_prediction(self):
        accumulator = ProbabilityAccumulator()
        for height in (4, 6, 8):
            accumulator.update_logits(logits_for(segmentation(height)))
        result = accumulator.result()
        torch.testing.assert_close(result.vcdr_samples[:, 0], torch.tensor([.4, .6, .8]))
        self.assertAlmostEqual(result.vcdr_scores()["vcdr_uncertainty"].item(), (0.08 / 3) ** .5)
        self.assertAlmostEqual(result.vcdr_scores()["vcdr_mean"].item(), .6)

    def test_ensemble_per_member_with_padding(self):
        masks = [segmentation(4), segmentation(6)]
        for mask in masks:
            mask[:, 0] = 2
        valid = torch.ones_like(masks[0], dtype=torch.bool)
        valid[:, 0] = False
        result = ensemble_predict([FixedModel(mask) for mask in masks], torch.zeros(1, 3, 12, 8), valid_mask=valid)
        self.assertAlmostEqual(result.vcdr_scores()["vcdr_uncertainty"].item(), .1)
        torch.testing.assert_close(result.vcdr_samples[:, 0], torch.tensor([.4, .6]))

    def test_primary_score_is_roi_bald_not_background_mean(self):
        probabilities = torch.zeros(1, 3, 2, 3)
        probabilities[:, 0] = 1
        probabilities[:, 0, 0, :2] = 0
        probabilities[:, 1, 0, 0] = 1
        probabilities[:, 2, 0, 1] = 1
        mi = torch.tensor([[[.2, .4, .9], [.9, .9, .9]]])
        result = UncertaintyResult(probabilities, probabilities * 0, mi, mi * 0, mi)
        score = result.image_scores()
        self.assertAlmostEqual(score["image_uncertainty"].item(), .3)
        self.assertEqual(score["image_uncertainty"].item(), score["bald"].item())
        self.assertEqual(score["roi_pixels"].item(), 2)
        # Background values cannot affect the score, even if nonfinite.
        mi[:, 1] = float("nan")
        mi[:, 0, 2] = 1000
        self.assertAlmostEqual(result.image_scores()["image_uncertainty"].item(), .3)
        # Padding must be excluded even when predicted as foreground.
        valid = torch.ones(1, 2, 3, dtype=torch.bool)
        valid[:, 0, 1] = False
        self.assertAlmostEqual(result.image_scores(valid)["image_uncertainty"].item(), .2)


if __name__ == "__main__":
    unittest.main()

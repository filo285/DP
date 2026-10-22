"""Shared probability summaries; no model-specific inference behaviour."""
from dataclasses import dataclass
import torch
from dp2.common.vcdr import valid_pixels, vertical_cup_to_disc, summarize_vcdr

SCORING_PROTOCOL = {
    "version": 2,
    "primary_image_score": "image_uncertainty = mean pixel mutual information in predicted disc (rim + cup)",
    "roi": "predicted foreground intersect valid pixels only; empty ROI gives NaN scores and is excluded from ranking",
    "vcdr_per_sample": "vertical cup extent / vertical full-disc extent from each pass/member argmax; padding excluded",
    "vcdr_summary": "vcdr_uncertainty = population standard deviation of per-sample vCDR; denominator N",
    "vcdr_invalid": "missing disc -> undefined ratio; any undefined sample or N<2 -> NaN uncertainty, not zero",
    "vcdr_absent_cup": "zero cup height -> ratio 0 if disc exists; missing-cup fraction is reported",
    "geometry": "full mask vertical extent (last row - first row + 1); no connected-component filtering or ellipse fit",
    "interpretation": "BALD and vCDR std are separate scores, not summed; no calibrated confidence intervals or empirically proven best ranking",
}


class ProbabilityAccumulator:
    """Welford moments over equally weighted predictions; population variance."""
    def __init__(self, valid_mask=None):
        self.count = 0
        self.mean = self.m2 = self.mean_entropy = None
        self.valid_mask = valid_mask
        self.vcdr_samples, self.disc_heights, self.cup_heights = [], [], []

    def update_logits(self, logits):
        if logits.ndim != 4 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
            raise ValueError("Expected finite B,C,H,W multiclass logits")
        p = logits.float().softmax(1)
        if self.mean is not None and p.shape != self.mean.shape:
            raise ValueError("All predictions must have the same shape")
        # Geometry is defined only for our three-class fundus segmentation.
        # Generic multiclass probability statistics remain usable independently.
        if p.shape[1] == 3:
            ratio, disc, cup = vertical_cup_to_disc(p.argmax(1), self.valid_mask)
            self.vcdr_samples.append(ratio)
            self.disc_heights.append(disc)
            self.cup_heights.append(cup)
        h = entropy(p)
        self.count += 1
        if self.mean is None:
            self.mean, self.m2, self.mean_entropy = p.clone(), torch.zeros_like(p), h.clone()
        else:
            delta = p - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (p - self.mean)
            self.mean_entropy += (h - self.mean_entropy) / self.count

    def result(self):
        if self.count < 2:
            raise ValueError("At least two predictions are required")
        predictive = entropy(self.mean)
        return UncertaintyResult(self.mean, (self.m2 / self.count).clamp_min(0).sqrt(),
                                 predictive, self.mean_entropy,
                                 (predictive - self.mean_entropy).clamp_min(0),
                                 vcdr_samples=torch.stack(self.vcdr_samples) if self.vcdr_samples else None,
                                 disc_heights=torch.stack(self.disc_heights) if self.disc_heights else None,
                                 cup_heights=torch.stack(self.cup_heights) if self.cup_heights else None,
                                 valid_mask=self.valid_mask)

def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)


@dataclass
class UncertaintyResult:
    mean_probabilities: torch.Tensor  # B,C,H,W
    std_probabilities: torch.Tensor   # population std, denominator T
    predictive_entropy: torch.Tensor  # B,H,W: H[E(p)]
    expected_entropy: torch.Tensor    # B,H,W: E[H(p)]
    mutual_information: torch.Tensor  # B,H,W: BALD
    vcdr_samples: torch.Tensor | None = None  # N,B: every pass/member
    disc_heights: torch.Tensor | None = None
    cup_heights: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None

    @property
    def prediction(self):
        return self.mean_probabilities.argmax(dim=1)

    def image_scores(self, valid_mask=None):
        """Average over predicted disc including cup; no ground truth needed.

        Empty predicted foreground gives NaN scores; never include background.
        valid_mask excludes letterbox padding (or other invalid input pixels).
        """
        pred = self.prediction
        valid = valid_pixels(pred, self.valid_mask if valid_mask is None else valid_mask)
        roi = (pred > 0) & valid
        empty = ~roi.flatten(1).any(1)
        count = roi.sum((1, 2))
        maps = {"std": self.std_probabilities.mean(1),
                "entropy": self.predictive_entropy, "bald": self.mutual_information}
        scores = {**{name: (value.masked_fill(~roi, 0).sum((1, 2)) / count.clamp_min(1)).masked_fill(empty, float("nan"))
                     for name, value in maps.items()},
                  "empty_foreground": empty, "roi_pixels": count,
                  "image_uncertainty_valid": ~empty}
        scores["image_uncertainty"] = scores["bald"]
        return scores

    def vcdr_scores(self):
        """Geometry always uses the valid mask supplied DURING inference."""
        if self.vcdr_samples is None:
            raise ValueError("vCDR requires three-class fundus predictions")
        prediction_ratio, _, _ = vertical_cup_to_disc(self.prediction, self.valid_mask)
        return {**summarize_vcdr(self.vcdr_samples), "vcdr_prediction": prediction_ratio,
                "vcdr_missing_cup_fraction": (self.cup_heights == 0).float().mean(0)}



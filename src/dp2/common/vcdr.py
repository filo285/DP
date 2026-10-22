"""Vertical cup/disc geometry and uncertainty for labels 0=bg, 1=rim, 2=cup."""
import torch


def valid_pixels(prediction, valid_mask=None):
    if prediction.ndim != 3:
        raise ValueError("Expected B,H,W class labels")
    valid = torch.ones_like(prediction, dtype=torch.bool) if valid_mask is None else valid_mask.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape or not valid.flatten(1).any(1).all():
        raise ValueError("valid_mask must match B,H,W and contain valid pixels per image")
    return valid


def vertical_cup_to_disc(prediction, valid_mask=None):
    """Measure vertical extents (last occupied row - first occupied row + 1).

    Disc includes rim AND cup. No component filtering or ellipse fitting: matches
    DP1's vCDR convention. Padding is excluded. Absent cup gives 0, absent disc
    gives NaN (undefined ratio), never 0. Heights use the input mask coordinates.
    """
    valid = valid_pixels(prediction, valid_mask)
    if not ((prediction >= 0) & (prediction <= 2)).all():
        raise ValueError("vCDR requires labels 0=background, 1=rim, 2=cup")

    def height(mask):
        occupied = mask.any(dim=2)
        rows = torch.arange(mask.shape[1], device=mask.device)[None, :]
        top = torch.where(occupied, rows, mask.shape[1]).amin(1)
        bottom = torch.where(occupied, rows, -1).amax(1)
        return (bottom - top + 1).clamp_min(0)

    disc_height = height((prediction > 0) & valid)
    cup_height = height((prediction == 2) & valid)
    ratio = cup_height.float() / disc_height.clamp_min(1)
    ratio = ratio.masked_fill(disc_height == 0, float("nan"))
    return ratio, disc_height, cup_height


def summarize_vcdr(samples):
    """Samples have shape passes/members x batch; std uses denominator N.

    A scalar uncertainty is valid only with >=2 samples and no undefined ratio.
    Mean over valid samples is diagnostic if some discs are missing. A missing
    disc is a quality failure, not evidence of low uncertainty.
    """
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("Expected nonempty N,B vCDR samples")
    finite = torch.isfinite(samples)
    count = finite.sum(0)
    values = torch.where(finite, samples, 0.)
    mean = values.sum(0) / count.clamp_min(1)
    deviations = torch.where(finite, samples - mean, 0.)
    std = (deviations.square().sum(0) / count.clamp_min(1)).sqrt()
    usable = finite.all(0) & (count >= 2)
    return {"vcdr_mean": mean.masked_fill(count == 0, float("nan")),
            "vcdr_uncertainty": std.masked_fill(~usable, float("nan")),
            "vcdr_valid_samples": count,
            "vcdr_invalid_fraction": 1. - count.float() / samples.shape[0],
            "vcdr_uncertainty_valid": usable}

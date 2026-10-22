# Whole-image uncertainty and vertical cup-to-disc ratio

Both MC dropout and deep ensembles use the same implementation in
`src/dp2/common/uncertainty.py` and `src/dp2/common/vcdr.py`. Model-specific
sampling remains in each method's own folder. No ground-truth masks enter scores.

## Primary image score: mean BALD in predicted foreground

For each pixel, compute `MI = H(mean(p)) - mean(H(p))` over stochastic passes or
ensemble members. Average this map over the final predicted disc, including the
cup (labels 1 and 2), excluding letterbox padding. This single number is exported
as `image_uncertainty`, with the existing `bald` column retained as an alias.
Higher scores mean more disagreement. If the predicted disc is empty, return NaN for all image uncertainty scores, set
`empty_foreground=True` and `image_uncertainty_valid=False`, and exclude the image
from the numeric ranking. Background and padding never enter the spatial average.

BALD is the selected default because it targets model disagreement, aligning
with the active-learning objective. Predictive entropy also includes uncertainty
shared by all predictions. Mean standard deviation and entropy remain available
as diagnostics. This is a methodological choice, not a claim that BALD has been
proven best on our validation data. See Gal et al. (2017), section 4:
https://proceedings.mlr.press/v70/gal17a/gal17a.pdf

For independent ensembles, MI is a BALD-style disagreement score, not exact
Bayesian posterior mutual information. The spatial average is a practical image
ranking heuristic, not joint mutual information over an entire segmentation.

## Anatomy score: standard deviation of vCDR

The vertical cup-to-disc ratio is

`vCDR = vertical diameter of optic cup / vertical diameter of full optic disc`.

The disc includes the rim AND the cup. This is a height ratio, not an area ratio.
Definition and segmentation context: Gao et al., *Automated vertical cup-to-disc
ratio determination from fundus images for glaucoma detection*:
https://pmc.ncbi.nlm.nih.gov/articles/PMC10891153/

For every individual dropout pass or ensemble member:

1. Take the argmax of its three class probabilities.
2. Exclude invalid/padded pixels.
3. Measure each structure's height as last occupied row minus first occupied row
   plus one, consistent with the original DP1 metric.
4. Save the heights and their ratio `r_t`.

The image-level anatomy uncertainty is

`vcdr_uncertainty = sqrt(mean((r_t - mean(r))**2))`.

This is population standard deviation (denominator N), matching our probability
STD convention. It stays in interpretable ratio units: samples 0.4 and 0.6 have
mean 0.5 and uncertainty 0.1. We use it instead of the standard error of the mean:
the objective is disagreement in the predicted anatomy, not the numerical error
in estimating the average. This aggregation is our implementation choice; the
vCDR source establishes the anatomical definition, not this acquisition policy.

`vcdr_mean` is the average of per-sample ratios. `vcdr_prediction` is the ratio
measured from the final mean-probability segmentation. They need not be equal.
Neither a large vCDR nor a large mean vCDR is by itself uncertainty.

An absent cup gives a zero ratio when a disc exists and is separately flagged by
`vcdr_missing_cup_fraction`. An absent disc gives an undefined ratio. If ANY pass
has an undefined ratio, `vcdr_uncertainty` is NaN and
`vcdr_uncertainty_valid=False`; it is excluded from numeric vCDR rankings and
listed in `vcdr_invalid.csv`. Review these failures separately; never treat NaN as
zero. The running std is also undefined until at least two samples exist.

We retain full mask extents without component filtering to match DP1. Isolated
predicted pixels can distort heights, and consistent incorrect shapes can have
zero vCDR std. The measure also misses shape changes that preserve both heights.
It complements BALD rather than replacing segmentation uncertainty. Values are
computed in the baseline's letterboxed coordinates, assuming an upright image;
uniform scaling cancels in the ratio, with remaining pixel quantization error.
Neither score is a calibrated confidence interval or a guarantee of correctness.

## Outputs and iterations

Each new inference run produces:

- `image_scores.csv`: one row per image with both scalar uncertainty scores,
  vCDR summaries, validity flags and evaluation-only Dice/vCDR error.
- `vcdr_iterations.csv`: one row per image per stochastic pass/member, with ratio,
  heights, running mean and running std. `iteration` is one-based; `sample_index`
  is zero-based. Ensemble member order matches `report.json`; MC seeds follow
  the recorded seed-plus-image-index rule.
- `ranking_bald.csv`: descending valid primary uncertainty, stable ties in input order.
- `image_uncertainty_invalid.csv`: images with no predicted foreground.

The per-pixel class distribution still includes the background class, so uncertainty
between foreground and background at a foreground pixel is retained. Exclusion
refers to background pixels in the spatial average, not removal of a class from
the probability distribution. Historical runs retain their original scoring policy;
new runs record scoring protocol version 2.

- `ranking_vcdr.csv`: descending vCDR uncertainty for valid images only.
- `vcdr_invalid.csv`: cases whose vCDR uncertainty cannot be computed.
- `image_*.npz`: pixel maps plus per-pass/member `vcdr_samples`, `disc_heights`
  and `cup_heights`.
- `report.json`: the scoring protocol, including invalid-case policy.

These iterations are inference samples, not training epochs or active-learning
rounds. Each future active-learning round can rerun this computation on its pool.
The current commands still evaluate validation data; rankings demonstrate the
scoring and do not move validation images into training. BALD and vCDR rankings
are separate alternatives, not a sum with arbitrary units or weights.

Example runs from the project root (choose unused output directories):

```powershell
.\.venv\Scripts\python.exe scripts/run_mc_dropout.py --limit 8 --passes 30 --output runs/mc_dropout/vcdr_pilot
.\.venv\Scripts\python.exe src/dp2/deep_ensembles evaluate --ensemble runs/deep_ensembles/full --limit 8 --output runs/deep_ensembles/vcdr_pilot
```

The ensemble command requires a completed training run. The existing
`runs/deep_ensembles/smoke_20261001` checkpoint can exercise the pipeline but is
not a trained research baseline. Old NPZ outputs do not contain per-sample
segmentations; rerun inference to obtain vCDR dispersion, rather than attempting
to reconstruct it from mean probabilities alone.

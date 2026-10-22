# DP2: uncertainty and active learning for fundus segmentation

Development workspace for the ORIGA/REFUGE experiments described in DP1.
Original datasets, notebooks and the trained baseline remain in
`C:/Users/PC1/Desktop/diplomovka`. Nothing in this setup retrains or overwrites them.

## Open and verify

Open this folder in VS Code. The configured interpreter is
`.venv/Scripts/python.exe`. In notebooks, select **Select Kernel > Python
Environments > .venv**; VS Code's notebook kernel selection is separate from the
Python interpreter setting.

From PowerShell, without needing to change execution policy:

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

Alternatively run the VS Code task **DP2: verify environment**.

## Dependencies

Python 3.13.3, PyTorch 2.10.0 with CUDA 12.8, and the direct dependency versions
match the existing system installation observed during setup. This records the
current machine, not proof of the exact historical DP1 training environment.

`requirements.txt` records direct dependencies. `requirements-lock.txt` captures
the complete installed environment after successful installation. To recreate:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-lock.txt
```

## Layout

- `configs/paths.json`: original dataset tables and baseline checkpoint location.
- `src/dp2/mc_dropout/`: dropout-specific inference and validation CLI.
- `src/dp2/deep_ensembles/`: independent training, checkpoint loading and inference CLI.
- `src/dp2/common/`: shared probability statistics and output generation.
- `src/dp2/baseline.py`: common DP1 architecture, preprocessing and metrics.
- `src/dp2/uncertainty.py`: compatibility imports for older notebooks only.
- `notebooks/`: exploration and visual analysis.
- `scripts/`: executable checks and future experiment entry points.
- `runs/`: future run outputs, excluded from Git.

Keep data, model checkpoints and `.venv` outside Git. Back up experiment outputs
separately. No remote repository is configured by this setup.

## MC-dropout uncertainty quantification

Both methods now export whole-image **mean BALD** (`image_uncertainty`) and
**standard deviation of per-pass/member vCDR** (`vcdr_uncertainty`). Per-iteration
ratios, running summaries and separate sorted rankings are saved automatically.
See [image scoring definitions and usage](docs/image_uncertainty.md).

Run the first UQ pilot using the existing trained baseline:

```powershell
.\.venv\Scripts\python.exe scripts/run_mc_dropout.py
```

This strictly loads the original checkpoint without downloading encoder weights,
freezes train/validation/test CSV manifests, and checks all 400 validation images
against the checkpoint's stored deterministic metrics (absolute tolerance 1e-5).
If parity fails, it saves the differences and stops before UQ. After parity passes,
it runs 30 stochastic passes on the first eight validation images in manifest order.
Only dropout is enabled; batch normalization stays frozen. The original DP1
preprocessing and mask conversion are preserved in `src/dp2/baseline.py`.

Outputs in `runs/mc_dropout/pilot/`:

- `report.json`: checkpoint hash, settings, environment details and parity results.
- `splits/*.csv`: frozen manifests with hashes in the report.
- `image_scores.csv`: predicted-region uncertainty scores and MC segmentation Dice.
- `image_*.npz`: mean class probabilities, population standard deviation,
  predictive entropy, expected entropy, BALD, segmentation and valid-pixel mask.
- `image_*.png`: visual summaries for the first eight images.

Use a new output directory for each run; existing output directories are rejected
to protect results. For the full validation set:

```powershell
.\.venv\Scripts\python.exe scripts/run_mc_dropout.py --limit 0 --passes 30 --output runs/mc_dropout/validation_30
```

Predictions are `argmax(mean(softmax(logits)))`. Standard deviation uses denominator
T, as in thesis equation 5.2, applied to probabilities rather than integer labels.
Predictive entropy is `H(mean(p))`; BALD is `H(mean(p)) - mean(H(p))`, in nats.
Predictive entropy includes multiple sources of uncertainty; BALD estimates
disagreement between stochastic predictions. These estimates are not calibrated
confidence intervals. See [MC dropout](https://proceedings.mlr.press/v48/gal16.html)
and [Bayesian active learning](https://proceedings.mlr.press/v70/gal17a.html).

Image scores average over the predicted disc including cup, excluding letterbox
padding. The std score first averages the three class standard deviations.
An empty predicted foreground gives NaN uncertainty and is excluded from rankings;
these cases are saved in `image_uncertainty_invalid.csv`. Background is never used. No ground-truth masks enter acquisition scores. Maps retain the baseline's
512-square letterboxed coordinates; use the saved valid mask when analyzing maps.
Seeds are recorded per image; reproducibility assumes the same software/device.

Run numerical and model-state checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This is a full-pool checkpoint UQ pilot. It does not demonstrate calibration or
active-learning gains. Uncertainty/error analysis across the full validation set and the
acquisition/retraining loop remain later experiments. Deep ensembles are implemented
separately as described below.
Do not initialise limited-label experiments from the full-pool trained checkpoint.


## Deep ensembles

Slovak walkthrough: [ensemble guide](src/dp2/deep_ensembles/README.md).

Train five independent members, then evaluate:

```powershell
.\.venv\Scripts\python.exe src/dp2/deep_ensembles train --members 5 --output runs/deep_ensembles/full
.\.venv\Scripts\python.exe src/dp2/deep_ensembles evaluate --ensemble runs/deep_ensembles/full --limit 8 --output runs/deep_ensembles/full_predictions
```

Each member gets a distinct seed, a fresh model, optimizer, shuffle order and
augmentation RNG, but identical frozen splits. By default the encoder starts from
ImageNet weights (may download); decoder/head initialization varies by seed. Use
`--encoder-weights none` for fully random initialization, and record that difference
when comparing with the pretrained MC baseline. No member starts from the trained
DP1 checkpoint. Training follows DP1's CE + Dice, AdamW, cosine scheduler, 60-epoch
maximum and patience 10, selecting each member by mean validation disc/cup Dice.
Members train sequentially; inference loads all members on the selected device.
Use `--device cpu` if device memory cannot hold the ensemble.

At inference each member runs once with dropout and batch normalization in eval
mode. Shared code averages probabilities and computes population std, entropy and
mutual information. CSV column `bald` denotes BALD-style ensemble disagreement;
independent members are not exact samples from a Bayesian posterior. The training
recipe uses the DP1 loss, without adversarial training; it is not an exact replication
of the [original deep ensembles paper](https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).

A quick end-to-end check uses two models, one epoch, eight training/four validation
images, random initialization and 128-pixel inputs:

```powershell
.\.venv\Scripts\python.exe src/dp2/deep_ensembles train --smoke --output runs/deep_ensembles/smoke_new
.\.venv\Scripts\python.exe src/dp2/deep_ensembles evaluate --ensemble runs/deep_ensembles/smoke_new --output runs/deep_ensembles/smoke_new_predictions
```

Smoke outputs are explicitly marked and are not comparable to the trained baseline.
`ensemble.json` records member seeds, checkpoint hashes and split hashes. Each
member has its own `best.pth` and `history.json`. Incomplete ensembles, duplicate
seeds, incompatible members and modified checkpoint files are rejected. Existing
output directories are never overwritten; interrupted training currently requires
a new output directory (resume is not implemented). These are full-pool experiments,
not limited-label active learning; test images are not used for checkpoint selection.

The legacy `scripts/run_uncertainty.py` command still invokes MC dropout. Historical
`runs/uq_mc_dropout` outputs remain untouched; new runs use the method subfolders.

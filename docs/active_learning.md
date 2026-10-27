# Active learning: complete experiment workflow

The project now supports frozen data preparation, simulated annotation reveal,
training on the current labeled subset, mask-free pool scoring, and separate
random/BALD/vCDR acquisition. MC dropout and deep ensembles are separate predictors.
The round controller, stage recovery, acquisition checkpoints, separate final test
evaluation, and evaluation graphs are implemented. Individual stage commands remain
available for inspection and component experiments.

## Configuration and data preparation

`configs/active_learning.json` freezes the experiment settings: five repetitions,
150 initial labels, six batches of 100, 30 MC passes, and five ensemble members.
The training settings reproduce the current recipe. Change a copy of this file before
preparation to define a different protocol; do not edit frozen outputs afterward.

From the project root in PowerShell:

```powershell
$alExperiment = "runs/active_learning/experiment_v1"
.\.venv\Scripts\python.exe scripts/run_active_learning.py prepare --output $alExperiment
```

Preparation reads the ORIGA and REFUGE tables configured in `configs/paths.json`.
It verifies image IDs, paths, file hashes, decoded-pixel hashes, and available patient
IDs before freezing the 1,050/400/400 population. Duplicate images or identified
patients crossing splits cause a failure with details in `audit.json`. Patient IDs
are dataset-qualified; cross-dataset identity cannot be established from these fields.
Near-duplicate visual similarity is not tested. If source tables have no patient
identifiers, the audit explicitly records that patient separation is unverified.

Each `repeat_XX/initial/` contains:

- `labeled.csv`: public metadata for the initial subset.
- `unlabeled.csv`: remaining training images, with no annotation columns.
- `labeled_annotations.csv`: revealed mask paths for exactly the labeled IDs.
- `state.json`: split membership hashes, counts, repetition, round, and smoke status.

The initial allocation is 93 ORIGA and 57 REFUGE images per repetition, sampled with
separate deterministic seeds. The private annotation tables contain mask paths;
image scoring and acquisition receive only public inputs. This is an enforced
application interface, not an operating-system access-control boundary. Preparation
checks that mask files exist but never reads their pixel contents.

`--data-root /path/to/data` can be supplied to each stage on another machine. It
changes the local path resolution while preserving relative paths, image IDs, and
frozen hashes. Keep the original relative directory layout. `prepare --paths ...`
supports another source-path configuration. Original absolute Windows CSV paths
are converted relative to `dp1_root`, including when preparation runs on Linux.

An audit of the actual data was completed in
`runs/active_learning/prepared_20261022/`: 1,850 images, no exact duplicates or split
overlap, and no patient IDs available. It also contains all five initial states.

## Run and resume a complete study

After preparing the experiment above, run its declared grid:

```powershell
$alStudy = "runs/active_learning/study_v1"
.\.venv\Scripts\python.exe scripts/run_active_learning.py run --experiment $alExperiment --output $alStudy
```

With the default research configuration, this runs both methods, all three strategies,
five repetitions, and matched full-pool references: 600 individual model fits after
sharing the initial predictors across strategies. This is the main research job.
For a functional check, first prepare a separate experiment with `prepare --smoke`;
its runner automatically uses the short smoke training settings. A smoke study still
scores the entire pool in its prepared manifests. No research gains can be inferred
from these short fits.

Optional `--methods mc_dropout`, `--strategies random bald vcdr`, and
`--repetitions 0` declare a smaller study grid before the first run. Keep the same
options when resuming. Missing random arms yield missing paired comparisons, not
invented baseline scores. Use a new study output for different settings.

```powershell
.\.venv\Scripts\python.exe scripts/run_active_learning.py run --experiment $alExperiment --output $alStudy --resume
```

The runner trains on accumulated labels, evaluates validation predictions, scores
the remaining pool, selects the configured batch, records an acquisition checkpoint,
and commits the next labeled/pool state. It stops after the fixed annotation budget,
including evaluation of the final trained predictor. Requests follow this schedule;
they are not triggered by an adaptive uncertainty threshold.

Completed stages are verified by hashes and reused. Finished ensemble members are
retained after an interrupted training stage; an unfinished member restarts with its
original seed. There is no mid-epoch optimizer resume. Stage outputs are first written
to a sibling `.partial` directory and published by an atomic rename. Interrupted work
is retained with an `.interrupted-...` suffix. Windows file-lock errors during directory
publication receive a bounded retry; a persistent error leaves recoverable artifacts.

An OS lock allows one writer per study directory and releases automatically when its
process exits. For concurrent jobs, use separate study output directories; each has
its own declared grid and summary. The controller executes members and arms sequentially
within one study. Keep device settings consistent when resuming unfinished training.
Use `--sequential` to reduce GPU memory during ensemble inference.

The study contains:

- `study.json`: immutable grid, configuration, and scoring/evaluation definitions.
- `shared/repeat_XX/<method>/`: round-zero predictor, validation, and pool scores
  shared by that method's strategies.
- `repeat_XX/<method>/<strategy>/round_XX/`: state, training, validation, selection,
  event, and round record. Later rounds have their own predictors and scores.
- `references/`: matched models trained on the full training population, with the
  same initialization/training recipe and repetition/member seeds used at round zero.
- `_complete.json` in each published stage: file integrity hashes.
- `progress.json` per arm, `last_error.json` when interrupted, and
  `study_complete.json` after the entire declared grid finishes.

At every request, `event/acquisition_checkpoint.json` records the exact requesting
model/member paths and hashes, per-member best/completed epochs and optimizer steps,
timestamp, label/pool counts before and after acquisition, selected IDs and scores,
fallback flags, seeds, and pre-acquisition validation metrics. Event IDs survive a
restart. `selected_images.png` shows up to six selected inputs without reading masks.
The terminal round has an explicit budget-completion record and no acquisition event.

## Validation summaries and acquisition graphs

```powershell
.\.venv\Scripts\python.exe scripts/run_active_learning.py summarize --experiment $alExperiment --study $alStudy --output runs/active_learning/summary_val_v1
```

For smoke studies, add `--include-smoke`; every graph is then labeled as a functional
check. Summaries use saved evaluation records and do not train models or rescore pools.

The output includes:

- `learning_curves.csv`: each repetition, method, strategy, round, label count,
  evaluation metrics, training/scoring time, epochs, checkpoint identity, and device.
- `acquisition_events.csv`: one row per request, linked to performance before the
  request and after the next trained round, with selected IDs and checkpoint details.
- `mean_curves.csv`: means, sample standard deviations, and counts across repetitions.
- `curve_areas.csv`, `paired_comparisons.csv`, and `paired_summary.csv`: normalized
  Dice curve areas and paired differences from random within a method/repetition.
- `fixed_budget_comparisons.csv`: per-metric paired differences at each label budget.
- `full_pool_references.csv`: separate empirical references, excluded from curve area.
- `figures/`: PNG/PDF label-budget graphs, per-run time graphs, and acquisition
  composition/invalid-score diagnostics.

Square markers on label-budget curves identify requests for new images; `+100`, for
example, marks an acquisition at the existing label count. The next performance point
appears only after training with those additional labels. Time graphs have vertical
request markers with round numbers. Elapsed wall time includes interruptions and idle
time; it is not averaged across repetitions. Training/scoring durations are recorded
separately, and shared initial fits are attributed to each arm for comparison.
Nonterminal time-curve points use the persisted acquisition timestamp, so resuming
an interrupted state publication does not move the requesting checkpoint on the graph.

Evaluation uses mean MC/ensemble probabilities before argmax, consistently with pool
scoring. Both legacy full-letterbox metrics and explicitly named `valid_*` metrics are
saved. The primary curve is mean valid-pixel disc/cup Dice. vCDR MAE is conditional on
defined reference/predicted ratios and is always accompanied by coverage and failure
counts. Hausdorff distances use resized/letterboxed pixel coordinates. Each evaluation
also retains `per_image.csv` for distributional analysis.

Curve area uses the complete common label interval specified in the frozen configuration
and trapezoidal integration divided by that interval's width. Bands show standard
deviation across repetitions, not confidence intervals; they are omitted for one
repetition. No target threshold was predeclared, so target-label efficiency is not
reported. Calibration and a downstream diagnostic classifier remain separate research
extensions, as specified in the implementation plan.

## Freeze the study and evaluate the held-out test set

Validation is used during the active-learning loop. Test labels are opened only by
the separate job below, after all declared arms and full-pool references are complete:

```powershell
.\.venv\Scripts\python.exe scripts/run_active_learning.py freeze --experiment $alExperiment --study $alStudy
.\.venv\Scripts\python.exe scripts/run_active_learning.py evaluate-study --experiment $alExperiment --study $alStudy
.\.venv\Scripts\python.exe scripts/run_active_learning.py summarize --experiment $alExperiment --study $alStudy --split test --output runs/active_learning/summary_test_v1
```

Freezing records hashes of the study and its completed artifacts. Changed or incomplete
studies are rejected. Test evaluation reads every saved round predictor, reuses shared
initial evaluations, and writes a separate result index. It does not modify acquisition
decisions, training checkpoints, or stopping rules. Test metrics are joined to the
previously recorded acquisition events when generating final graphs. The test job can
be rerun after interruption to reuse completed evaluation stages.

## Train a selected subset manually

```powershell
$alState = "$alExperiment/repeat_00/initial"
$alModel = "runs/active_learning/repeat00_mc_initial"
.\.venv\Scripts\python.exe scripts/run_active_learning.py train --experiment $alExperiment --state $alState --method mc_dropout --output $alModel
```

Use `--method deep_ensembles` and a new output directory for the ensemble arm.
Both use `dp2.common.training.train_model`, the same trainer now used by the
standalone full-pool ensemble command. Each fit constructs a fresh model, optimizer,
and scheduler. No DP1 or previous-round checkpoint is loaded for initialization.
The encoder may use ImageNet weights, which PyTorch may download on first use;
configure `TORCH_HOME` on school machines if a shared or pre-populated cache is needed.

Training reads masks only for the labeled subset and validation images. Test masks
and unlabeled training masks are not read. Model selection uses the existing mean
validation disc/cup Dice rule. MC and the first ensemble member use a matched seed;
other members have distinct seeds. Seeds depend on repetition, round, and member,
so strategies can be compared with matched training randomness.

Saved outputs include `best.pth`, `history.json`, and `training.json` per member,
plus `model.json` or `ensemble.json`. They record labeled-set provenance, split hashes,
seeds, selected epoch, completed epochs, optimizer steps, initialization, and timings.
The active-learning loader rejects full-pool models, incompatible members, modified
checkpoints, and models trained on a different labeled/pool state. Historical DP1
parity checking remains in the original MC pilot command.

## Score the remaining pool

```powershell
$alScores = "runs/active_learning/repeat00_mc_scores"
.\.venv\Scripts\python.exe scripts/run_active_learning.py score --experiment $alExperiment --state $alState --predictor $alModel --output $alScores
```

The predictor manifest determines whether MC dropout or deep ensembles are used.
For ensembles, add `--sequential` to retain members on CPU and transfer one at a
time to the selected GPU. This reduces GPU memory use at the cost of transfers.
Use `--device cpu` to run either method on CPU.

The scorer uses the existing uncertainty implementation unchanged: scoring protocol 2,
mean BALD inside predicted disc/cup excluding padding, and population standard
deviation of per-sample vCDR. The valid-pixel mask comes from image dimensions only.
The compact `pool_scores.csv` contains no reference segmentation or accuracy metrics.
`report.json` links scores to predictor checkpoints, pool membership, settings, seeds,
timing, and invalid-score counts. MC seeds use stable image IDs, not pool row indices.

## Select images and reveal annotations

```powershell
$alSelection = "runs/active_learning/repeat00_mc_bald_selection"
.\.venv\Scripts\python.exe scripts/run_active_learning.py select --experiment $alExperiment --state $alState --method mc_dropout --strategy bald --scores $alScores --output $alSelection

$alNextState = "runs/active_learning/repeat00_mc_bald_state01"
.\.venv\Scripts\python.exe scripts/run_active_learning.py reveal --experiment $alExperiment --state $alState --selection $alSelection --output $alNextState
```

For the vCDR arm, use `--strategy vcdr` and a different output/next-state directory,
starting from the same initial state. Initial scores may be reused across strategies
because the initial predictor is shared. After their selections differ, train and score
each arm separately. Run the ensemble strategies using their own ensemble scores.

For random selection, use `--strategy random` and omit `--scores`. Random acquisitions
use matching seeds across MC and ensemble arms within a repetition. The `--count`
option overrides the batch size for an explicit component experiment; use the frozen
default for the planned main study. Selection never exceeds the remaining pool size.

Uncertainty selection ranks valid scores in descending order, with seeded random
tie-breaking independent of input order. If too few valid scores remain, it fills
the batch by random selection from invalid cases and flags them as fallbacks.
An entirely invalid ranking fails with a saved report and reveals no labels.

`ranking.csv` contains the full ranking and `acquisition.csv` the selected rows.
`selection.json` records method, strategy, seed, timestamp, provenance, score source,
checkpoint references, requested/actual counts, fallback counts, and file hashes.
The reveal stage verifies those records and creates a new immutable state; prior
states are preserved. Selecting an already labeled, validation, or test image is
rejected. Subsequent training uses all accumulated labeled images in the new state.

These manual stage outputs can be inspected independently. Use `run` for the complete
controller and the acquisition-event/graph workflow described above.

## Smoke checks and tests

Use `prepare --smoke --output <new-directory>` for eight initial labels, two-image
batches, two passes/members, and one repetition. Training this protocol automatically
uses 128-pixel images, one epoch, batch size two, random initialization, and at most
four validation images. The full training population remains frozen and available
to the image-only pool scorer.

Alternatively, `train --smoke` uses the same short training settings on all currently
labeled images in an existing state. Smoke checkpoints and selections are flagged;
revealing a smoke-derived selection propagates that status to later states and fits.
Smoke results must not be used as research results. Manual stage output directories
must be new; the study runner supports recovery through `run --resume`.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover pool scoring with physically removed masks, selected/validation-only mask
reads during training, deterministic replay, separate BALD/vCDR rankings, invalid-score
handling, duplicate and patient-overlap rejection, portable roots, subset provenance,
state tampering, manual selection/reveal, and rejection of stale pool scores.
The existing uncertainty and ensemble tests remain part of this command.

Verification completed on 27 October 2026: all 31 tests passed, including full-grid
execution, recovery after selection/state publication, completed-member reuse,
temporary Windows directory locks, test isolation, artifact tampering, and graph joins.
A CUDA smoke study with the actual ResNet34 U-Net used 16 copied ORIGA/REFUGE training
images: 12 for its training pool, two for validation, and two for its miniature test.
Both methods and all three strategies completed two acquisitions (8 -> 10 -> 12 labels),
producing 12 acquisition events, 18 round records, and two full-pool references.
Frozen test evaluation and summaries completed. The original held-out test masks were
not accessed. This one-epoch check is not an accuracy experiment.
Artifacts are in `runs/active_learning/real_study_smoke_20261027/`; the visually reviewed
graphs and exports are in `test_summary_reviewed/`. The full research study has not run.

Verification completed on 22 October 2026: all 24 tests passed. Real-data one-epoch
smoke training completed for one MC model and two ensemble members on the 150-image
initial subset. The ensemble scored all 900 pool images; each acquisition strategy
selected 100 images, and BALD reveal produced a 250-labeled/800-unlabeled smoke state.
MC scoring was additionally checked on eight real images with two passes, and
sequential/resident ensemble GPU outputs agreed on two images. The original standalone
ensemble smoke command also passed after extraction of the shared trainer.
Machine-readable checks are saved in `runs/active_learning/verification_20261022.json`.
These runs verify functionality only. Seeded GPU training can still differ numerically
between hardware/software environments; retain the environment and checkpoint records.

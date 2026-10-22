# Active learning implementation plan

Prepared on 20 October 2026 after reviewing the current project, training code,
uncertainty code, dataset tables, and saved experiment reports.

## 1. Objective and agreed scope

Implement a reproducible simulated active learning experiment for optic disc/cup
segmentation. Determine whether selecting images using uncertainty improves
segmentation and vCDR accuracy with fewer training annotations than random selection.

User-confirmed decisions:

- Simulate annotation using existing masks, revealed only after selection.
- Compare MC dropout and deep ensembles.
- For each method, compare random selection, mean foreground BALD, and vCDR
  standard deviation as three separate acquisition strategies.
- School computers are available; compute is not the main scope constraint.
- Log explicit checkpoints whenever a run requests new images, and include those
  acquisition events in evaluation graphs.

The numerical schedule below is a proposed default, not an existing project setting.
Freeze it after the pilot and before the main experiments. The first implementation
evaluates segmentation and anatomical measurement; a glaucoma classifier or a manual
annotation interface would be a separate extension.

## 2. Starting point and implementation gaps

Available components:

| Component | Current implementation | Planned use |
|---|---|---|
| Architecture, masks, transforms, metrics | `src/dp2/baseline.py` | Preserve DP1 conventions |
| Training recipe | `src/dp2/deep_ensembles/train.py` | Extract a reusable single-model trainer |
| MC dropout | `src/dp2/mc_dropout/inference.py` | Score and evaluate a model trained on the current labeled subset |
| Ensemble prediction | `src/dp2/deep_ensembles/inference.py` | Score and evaluate independently trained members |
| Probability and vCDR statistics | `src/dp2/common/uncertainty.py`, `vcdr.py` | Reuse both acquisition scores without changing their definitions |
| Existing exports | `src/dp2/common/export.py` | Keep for supervised inspection; build a separate mask-free pool scorer |

Important gaps found in the code:

1. Ensemble training currently rebuilds the complete training split internally.
   It needs an explicit labeled-manifest input for each active learning round.
2. `FundusSegDataset` always loads a reference mask. The existing export also
   computes ground-truth Dice and vCDR error. Neither should be the pool-scoring path.
3. The MC CLI always loads the original full-pool DP1 checkpoint and performs
   historical validation parity checks. Active learning needs a loader for each
   round's newly trained checkpoint, without the DP1 parity requirement.
4. Pool updates, annotation reveal, round orchestration, recovery, and learning-curve
   aggregation do not yet exist.
5. The inspected ensemble reports refer to a two-member, one-epoch smoke run.
   Those outputs establish functionality, not a research-quality comparison.

The full-pool DP1 model must never initialize a limited-label run. Its training
history already includes the images whose labels the experiment is meant to hide.

## 3. Experimental protocol

### Data and budget

Preserve the current top-level split unless the overlap audit finds a problem:

- Training population: 1,050 images, comprising 650 ORIGA and 400 REFUGE training images.
- Validation: the existing 400 REFUGE validation images.
- Test: the existing 400 REFUGE test images.
- Within each repetition, sample 150 initially labeled training images and keep
  the other 900 in the unlabeled pool.
- Acquire 100 images per round for six rounds.
- Evaluate at 150, 250, 350, 450, 550, 650, and 750 training labels.
- Add a separately trained 1,050-image supervised reference for each method.

Initial sampling should preserve the training population's dataset proportions
approximately: 93 ORIGA and 57 REFUGE images. Use only dataset identity and random
sampling; do not balance using hidden glaucoma labels, mask-derived vCDR, or mask areas.
Acquisition itself remains unrestricted across the remaining training population.

These are training-label budgets. Validation requires another 400 labeled images,
and test evaluation requires 400 more. Report those separately so annotation-efficiency
claims do not conceal the fixed evaluation-label cost. Image counts represent annotation
units, not measured expert annotation time.

### Fair comparisons

- Run five independent repetitions, with the initial subset and acquisition RNG
  independently sampled for each repetition.
- Within a repetition, all strategies and both methods start with the same labeled IDs.
- Freeze validation/test membership across rounds and repetitions.
- Use identical architecture, preprocessing, training objective, initialization source,
  and checkpoint-selection rule across acquisition strategies.
- Use one model for MC dropout, 30 stochastic inference passes, and five independently
  trained members for ensembles. Keep dropout off during ensemble inference.
- Reuse the round-zero model/ensemble across acquisition strategies within a method
  and repetition. After the first acquisition, each strategy follows its own history.
- Use a shared random-selection trajectory across methods in a repetition; the
  predictors still train separately. Never give a method a more favorable random baseline.
- Define independent deterministic seed streams for initial sampling, training,
  acquisition, and MC inference. Training seeds depend on repetition, round, and member,
  rather than the strategy, to support paired comparisons.

### Training between rounds

Recommended main protocol: train fresh models on the entire accumulated labeled
subset at every round. Restart the optimizer and scheduler. Initialize the encoder
from the same ImageNet weights and initialize the remaining layers using the recorded
seed. For ensembles, use distinct seeds for members.

This makes the round's labeled set the main changing factor and avoids a warm-start
history confound. Continuing from the previous round can be a later, separately reported
compute-saving ablation; do not mix the two protocols in the main comparison.

Preserve the current recipe initially: ResNet34 U-Net, dropout 0.2, 512-pixel letterboxing,
CE plus multiclass Dice, AdamW, learning rate 1e-4, weight decay 1e-4, cosine schedule,
maximum 60 epochs, and patience 10. Select checkpoints by mean validation disc/cup Dice.
Keep these choices fixed across strategies after the pilot.

## 4. Implementation sequence

### Phase 1 — Freeze and audit the experiment population

Create `src/dp2/active_learning/protocol.py` and `state.py`, with a versioned
`configs/active_learning.json` and an experiment-preparation command.

- Freeze explicit train, validation, and test manifests; record hashes and source paths.
- Introduce stable, dataset-qualified image IDs; filenames alone can collide across datasets.
- Audit duplicate IDs, resolved image paths, image contents, and split intersections.
  Review suspected near-duplicates if exact hashes do not settle their identity.
- Search available source metadata for patient/group identity and prevent patient
  overlap where it can be established. The inspected CSV headers have no explicit
  patient/subject identifier; eye side is not a patient identifier. If no mapping is
  available, record that patient independence could not be verified.
- If grouping requires a revised split, version it as a new protocol and regenerate
  all comparison runs and supervised references consistently.
- Store only allowed image metadata in public pool manifests. Keep mask paths,
  expert CDR, glaucoma labels, mask areas/heights, and other annotation-derived
  columns in the annotation store, inaccessible to the selector.

Acceptance: labeled and unlabeled IDs are disjoint, their union is exactly the
training population, and neither overlaps validation/test. Audit conclusions are saved.

### Phase 2 — Add image-only loading and controlled annotation reveal

Create `data.py` and `annotation.py` under `src/dp2/active_learning/`.

- Implement an image-only pool dataset returning image ID, transformed image, and
  valid-pixel mask. Reuse evaluation normalization and letterboxing.
- Generate the valid mask from image geometry, never from reference segmentation.
- Implement a simulated annotation provider accepting selected IDs and returning
  their existing reference mask paths. Training receives only revealed records.
- Create the initial labeled set through the same reveal mechanism.
- The selector receives IDs, allowed metadata, scores, and validity flags only.
  It must not receive reference masks or per-image accuracy measurements.

Acceptance: score a pool whose mask files are absent or inaccessible, and verify that
no mask-loading function is called. Revealed IDs must match the committed selection.

### Phase 3 — Make training reusable for labeled subsets

Extract training into `src/dp2/common/training.py`, then adapt the existing ensemble
command to call it. The callable accepts explicit training/validation manifests,
configuration, seed, output directory, and provenance.

- Support one independently trained model and multiple independent ensemble members.
- Record the exact labeled-ID hash in each checkpoint and ensemble manifest,
  alongside the global split hashes, initialization, round, and repetition.
- Reject checkpoints whose labeled-set provenance does not match the round.
- Never load the full-pool DP1 checkpoint in this path.
- Preserve the current standalone ensemble CLI behavior through the shared trainer.
- Cache encoder weights on school machines and configure dataset roots independently
  of the current Windows paths. Resolve stable IDs to local paths without changing
  the experiment's logical data membership.

Acceptance: train on a small specified subset and verify that no other training masks
are read; produce both a single-model checkpoint and a compatible two-member smoke ensemble.

### Phase 4 — Add pool scoring and selection

Create `scoring.py` and `acquisition.py`.

Scoring adapters should call the existing MC or ensemble prediction functions and
write compact tables rather than full maps for every image in every round.

| Strategy | Selection rule |
|---|---|
| Random | Uniform sampling without replacement from all remaining pool IDs |
| BALD | Descending mean pixel mutual information in predicted disc including cup, excluding padding |
| vCDR uncertainty | Descending population standard deviation of per-pass/member vertical cup-to-disc ratios |

Preserve scoring protocol version 2. BALD and vCDR are independent alternatives;
do not sum them. Ensemble BALD is a disagreement heuristic with the same mathematical
form, not exact Bayesian posterior mutual information. Pixel-averaged segmentation
BALD is also an image-ranking heuristic rather than joint segmentation information.

Handle ties and invalid values explicitly:

- Break equal finite scores with a seeded random tie order, avoiding filename/dataset
  ordering effects. Save the tie seed and final ranking.
- Keep NaNs invalid; never replace them with zero.
- Rank valid scores first. If fewer than the requested batch are valid, fill the
  remaining slots uniformly from invalid-score IDs and flag each fallback selection.
- If all pool scores are invalid, stop the research run with a failure report for
  inspection rather than silently turning the strategy into random sampling.
- Report invalid-score rates every round. Diagnostic rankings may still omit invalid
  scores; the acquisition manifest must describe its separate fallback policy.
- Base MC inference seeds on stable image IDs, repetition, and round, not a shrinking
  pool's row index. Record the seed mapping.

For ensemble memory limits, support sequential member inference with the existing
probability accumulator, or allocate an appropriate school GPU. Avoid treating CPU
fallback as the only solution to holding all five members in memory.

Acceptance: deterministic replay selects the same unique pool IDs, fills the configured
batch where possible, and never uses evaluation columns or hidden annotations.

### Phase 5 — Implement the round controller and recovery

Create `runner.py`, `cli.py`, and `scripts/run_active_learning.py`.

For round r:

1. Validate and load the committed labeled/unlabeled state.
2. Train fresh model(s) using only the current labeled set; select checkpoints on validation.
3. Save validation measurements using the method's prediction rule.
4. If the annotation budget is reached, finish without another acquisition.
5. Score only the remaining training pool, or directly sample for the random strategy.
6. Select IDs and save an immutable acquisition manifest and acquisition checkpoint
   before revealing those masks.
7. Commit the next state's labeled/unlabeled manifests atomically.

At every request for new images, retain the exact selected model checkpoint (or
all ensemble member checkpoints) that produced the acquisition scores. Save an
`acquisition_checkpoint.json` referencing their paths and hashes; model weights need
not be duplicated if the referenced artifacts remain immutable and available.

Record the following for each acquisition event:

- Unique event ID, experiment ID, repetition, method, strategy, source round, and
  destination round.
- UTC request timestamp and elapsed experiment time, with explicit units.
- Training completion epoch, selected checkpoint epoch, optimizer-step count, and
  cumulative training steps. For ensembles, retain per-member values rather than
  inventing a single common epoch across independently stopped members.
- Labeled/pool counts before the request, requested batch size, selected IDs and
  their scores/validity/fallback flags, and expected counts after annotation reveal.
- Hashes of the current labeled/pool manifests, scoring protocol, seed settings,
  and acquisition manifest.
- Reference to pre-acquisition validation metrics from the requesting predictor,
  including disc/cup Dice, vCDR MAE, and failure coverage. Once the next round is
  trained, join its evaluation through the event ID to show performance after adding
  the requested images. Keep that later measurement in a separate evaluation record.

The request event occurs after training and selection, before annotation reveal.
Under the proposed protocol it is triggered by the fixed round schedule; logging
does not introduce an uncertainty threshold or a new adaptive stopping rule. Record
budget completion as a distinct terminal event, without a request for more images.

Use explicit states such as prepared, training, trained, scored, selected, and committed.
Restarting must never select a second batch or reveal extra labels for the same round.
Event IDs must remain stable across restarts so recovered runs cannot duplicate
acquisition markers. Build an experiment-level `acquisition_events.csv` from committed
per-event records for plotting and analysis.
Initially support recovery from completed stages and restart an interrupted member's
training with its original seed. Exact mid-epoch continuation would require optimizer,
scheduler, scaler, sampler, and RNG state; add it only if operationally needed.

Output layout:

```text
runs/active_learning/<experiment>/
  protocol.json
  splits/{train,val,test}.csv
  audit.json
  repeat_00/
    initial_labeled.csv
    mc_dropout/bald/round_00/
      state.json
      labeled.csv
      unlabeled.csv
      training/
      validation_metrics.json
      pool_scores.csv
      acquisition.csv
      acquisition_checkpoint.json
    ...
  summary/
    learning_curves.csv
    acquisition_events.csv
    paired_comparisons.csv
    figures/
```

Random runs may omit pool scores. Save selected visual examples and small diagnostic
samples; enable dense map export explicitly to control storage use.

Acceptance: two successive acquisitions complete end to end, and interruption after
selection or during state commit does not change the resulting labeled population.
Each request resolves to the exact requesting predictor and appears exactly once
in the aggregated acquisition-event table.

### Phase 6 — Add evaluation and experiment summaries

Create `evaluation.py` and `analysis.py`.

- Evaluate the method actually used: mean probabilities over 30 MC passes or five
  ensemble members, followed by argmax segmentation. For a deterministic baseline,
  report it separately; the existing single-model `evaluate()` alone is insufficient
  for the principal MC-versus-ensemble learning curves.
- Preserve legacy DP1 metric reporting for comparability and add explicitly named
  valid-pixel metrics for the active learning experiment. Apply the same geometric
  convention to all methods and record any difference from historical numbers.
- Primary outcome: normalized area under the mean disc/cup Dice learning curve over
  the common training-label interval 150–750, using trapezoidal integration.
- Also report disc Dice, cup Dice, vCDR MAE, per-image distributions, and prediction
  failures. Keep Hausdorff as a secondary metric with its coordinate convention stated.
- Do not silently discard undefined vCDR: report missing-disc counts and valid-case
  coverage next to conditional MAE. Define any failure-penalty sensitivity analysis
  in the protocol before comparing strategies.
- Report differences from random within each repetition and method, plus all five
  individual curves and mean/standard deviation. Treat repetitions, not pixels, as
  the units of variation for strategy comparisons. Five repetitions support an
  exploratory estimate; avoid strong significance claims from such a small sample.
- Report labels required to reach a predeclared performance target, if a target is
  chosen before final evaluation. Mark unreached targets explicitly and report any
  interpolation rule. Otherwise use curve area and fixed-budget comparisons.
- Include wall-clock training/scoring time, epochs, and GPU details. Equal annotation
  budgets do not imply equal compute between MC dropout and ensembles.
- Mark every request for new images on learning curves, linked to its acquisition
  event ID. On label-count graphs, place the request marker at the current labeled
  count and pre-acquisition performance; annotate the batch size (for example,
  `250 labels: request +100`). Show the next trained predictor's result at 350 labels.
  Revealing labels alone must not imply an immediate performance improvement.
- Also generate per-run performance-versus-elapsed-time graphs with vertical request
  markers, round numbers, and labeled counts. Use cumulative steps or round boundaries
  for optional training-progress plots; epoch counters restart with fresh training,
  and ensembles have separate member histories. Do not align independent runs by
  wall-clock timestamps when averaging their learning curves; use common label budgets.
- Export graph-ready event/evaluation tables so each plotted point can be traced to
  saved weights, label counts, and metrics. Include the initial evaluation and final
  budget evaluation, explicitly distinguishing them from acquisition requests.
- Plot selected dataset composition and invalid-score rates to help interpret
  acquisition behavior. Hidden annotations may be inspected retrospectively only
  after acquisition decisions are finalized; never feed that analysis back into selection.

Validation guides checkpoint selection and pilot decisions. Freeze the complete
protocol before a separate final evaluation job loads test masks. That job may evaluate
every saved round checkpoint to produce test learning curves, but test results must not
feed back into acquisition, stopping, model selection, or a revised protocol.
Join those retrospectively computed test metrics to the saved event/checkpoint IDs;
test metrics are not computed or stored as inputs at acquisition time.

Train the full-pool reference with the same recipe, seeds, initialization, prediction
rule, and evaluation conventions as the corresponding method. It is an empirical
reference, not a guaranteed performance upper bound. Historical DP1 results remain
context rather than a substitute for these matched references.

Calibration and a downstream diagnostic classifier appear in the wider thesis scope.
If required for the final thesis, add a separately specified evaluation work package;
the three acquisition comparisons alone do not establish calibration or diagnostic benefit.

Acceptance: a single summary command generates paired curves and comparison tables
from immutable reports without retraining or recomputing acquisitions.

## 5. Pilot and full experiment execution

1. **Logic smoke test:** tiny training/pool subsets, two ensemble members, two MC
   passes, 128-pixel images, one epoch, and two acquisition steps. Mark all outputs
   as smoke-only and prevent them entering research summaries.
2. **Research pilot:** one repetition at 512 pixels, 150 initial labels, and one
   100-image acquisition for all three strategies and both methods. Use the actual
   30-pass/five-member settings. Inspect learning, invalid predictions, provenance,
   runtime, and successful restart behavior using validation only.
3. **Freeze protocol:** confirm or adjust the label schedule and training settings
   from the pilot; record all changes before the main study.
4. **Main study:** five repetitions, both methods, three strategies, six acquisitions.
   Parallelize independent repetitions/strategies or ensemble members on school machines.
   Use unique output ownership and frozen manifests to avoid competing writes.
5. **Matched full-pool references and final test evaluation:** complete the references,
   then run the frozen test-evaluation and aggregation commands.

With the proposed settings and round-zero sharing, the main study needs
`5 repetitions × (1 MC model + 5 ensemble members) × (1 shared initial fit + 3 strategies × 6 later fits)`
= **570 individual model trainings**, plus **30** full-pool reference trainings.
These are training-job counts, not GPU-hour estimates; benchmark actual durations
in the pilot. Pilot jobs are additional unless their frozen protocol and provenance
match the final study exactly.

## 6. Verification and completion criteria

Add targeted tests for the new behavior:

- Split/ID overlap and forbidden annotation-column rejection.
- Pool scoring without readable masks; reveal only selected labels.
- Labeled-set checkpoint provenance and rejection of full-pool initialization.
- Stable seeds, tie handling, NaNs, fallback selection, and shrinking pools.
- Exact pool accounting: no duplicate acquisition, no validation/test acquisition,
  monotonically growing labeled sets, and correct final budget.
- Recovery after training, selection, and interrupted state commits.
- Exactly one acquisition checkpoint per request, resolvable model/member hashes,
  correct pre/post label-count associations, and graph markers matching logged events.
- Correct method-specific evaluation and explicit invalid-vCDR coverage.

Run the existing uncertainty, geometry, inference-mode, and checkpoint tests after
extracting shared training code. Keep mathematical scoring behavior unchanged unless
introducing a separately versioned experimental protocol.

The implementation is complete when one configured command can run either method
with any of the three strategies, resume safely at stage boundaries, produce auditable
per-round outputs, and generate paired annotation-efficiency curves. An observed gain
over random is an experimental finding, not a software completion requirement.

## 7. Methodological sources

- [Gal, Islam, and Ghahramani (2017), Deep Bayesian Active Learning with Image Data](https://proceedings.mlr.press/v70/gal17a.html):
  foundation for uncertainty-based image acquisition. This project's foreground
  segmentation aggregation and vCDR acquisition are project-specific choices.
- [Gal and Ghahramani (2016), Dropout as a Bayesian Approximation](https://proceedings.mlr.press/v48/gal16.html):
  background for MC dropout uncertainty estimation.
- [Kirsch, van Amersfoort, and Gal (2019), BatchBALD](https://proceedings.neurips.cc/paper/2019/hash/95323660ed2124450caaac2c46b5ed90-Abstract.html):
  evidence that independently ranking uncertain points can select redundant batches.
  Retain simple top-score batching for the agreed first comparison; consider diversity
  as a later extension and do not call this implementation BatchBALD.

# Experiment log

Fixed split: `split.json` (6,791 val breaths, 10%, stratified by R×C, seed 42). Metric: MAE on
`u_out == 0` rows. Machine-readable table: `results.tsv`. "Snapped" = predictions rounded to the
nearest of the 950 discrete pressure values seen in train.

## What the data looks like (from our own EDA, 2026-11-10)

- 67,905 train breaths, 7,545 test breaths, every breath exactly 80 rows, no nulls, no breath
  overlap between train and test. 9 lung settings (R ∈ {5,20,50} × C ∈ {10,20,50}); R=50,C=10 is
  over-represented (~18%).
- 38% of rows are inspiratory (25–32 per breath, always the first rows; `u_out` never returns to 0).
- `time_step` spacing is irregular (median 0.033 s, up to 0.25 s), so `dt` is a feature.
- `pressure` takes only 950 distinct values on an even 0.0703 grid (sensor quantization) → we can
  snap predictions to that grid.

## Trials

| # | Trial | Val MAE | Snapped | Runtime | Device | What it tested → what we learned |
|---|-------|---------|---------|---------|--------|----------------------------------|
| 0 | `gbm_baseline` | 0.7808 | 0.7804 | 2.5 min | CPU (M2 Pro) | Per-row HistGradientBoosting (L1) on 34 features (u_in lags/leads/diffs, cumulative u_in, dt, R/C one-hots). → Valid pipeline. Error is worst for R=50 (0.92–1.16) and C=10. A per-row model can't carry the breath's history; next: a sequence model. |
| 1 | `smoke` | 0.7481 | 0.7478 | 5.1 min | MPS | 4-layer BiLSTM (h=256), L1 loss on inspiratory rows, 5 epochs, raw-scale target. → Runs at ~1 min/epoch on the Mac. Epoch-1 train MAE was 5.3: the net spent epochs learning the pressure offset. Next: standardize the target. |
| 2 | `lstm_v1` | **0.2446** | **0.2435** | 42.2 min | MPS | Same model + output rescaled by train pressure mean/std, 40 epochs, one-cycle LR 2e-3. → 3.2× better than the baseline. Val was 0.353 at epoch 25 and still falling at the end (train 0.204 vs val 0.245), so it is under-trained, with a train/val gap starting to open. Snapping gains ~0.001. Worst settings: R=50,C=20 (0.358) and R=50,C=50 (0.353); best: R=5,C=20 (0.148). |

Mid-run crash worth remembering: predicting with batch 1024 ran out of MPS memory; fixed with
batch 256 + `torch.mps.empty_cache()`.

## Current best

`lstm_v1` → `submission.csv` (603,600 rows, validated against `sample_submission.csv`).
Not yet submitted to AI Coding Gym.

## Next hypotheses (ordered by expected value per GPU-hour)

1. **Train longer.** v1 was still improving at epoch 40. On the 1080 Ti try 150 epochs, same
   config. Confirms if val drops well below 0.20; killed if val plateaus while train keeps falling
   (then regularize instead).
2. **Close the train/val gap** if (1) shows one: dropout 0.1 between LSTM layers, or a small
   weight decay.
3. **Error analysis of the best model**: val error by time step within the breath and by R×C.
   R=50 is ~1.5–2× worse than R=5; check whether the error is concentrated at the start of the
   breath or the pressure peak, and add a targeted feature if so.
4. **Auxiliary loss on expiratory rows** (`--exp_weight 0.1–0.5`): the unscored rows are free
   supervision about the same lung dynamics.
5. **Capacity**: hidden 384 or 512, only after (1) shows the 256 model saturating.
6. **Seeds + blend**: 3 seeds of the best config, median blend via `make_submission.py`. Then
   refit on train+val (`--full`) for the final submission.

## Disclosures (course rule: flag competition-specific ideas)

- The assistant (Claude) that wrote this pipeline was trained on public internet text and has
  general familiarity with the fact that this was a public Kaggle competition. No solutions were
  looked up, and none were consulted during this work.
- Choices and where they came from: cumulative `u_in` (physics: lung volume is the integral of
  flow, P ≈ R·flow + V/C); bidirectional LSTM (generic choice for fixed-length sequences where the
  whole input is known up front); L1 loss on inspiratory rows (it is the metric); grid snapping
  (found in our EDA above: 950 distinct pressure values); median blending (generic for MAE).
  These are generic or evidence-derived, but they also overlap with approaches widely used in that
  competition, so they are flagged here for the report.

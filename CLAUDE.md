# PA3 — Ventilator Pressure Prediction (DSC 102, AI Coding Gym)

Predict airway `pressure` for every row of `data/test.csv`. Score = MAE on inspiratory rows
(`u_out == 0`) only. Lower is better. Read `EXPERIMENTS.md` first: it holds every trial so far and
what to try next.

## Hard rules (course policy — do not break these)

- Use **only** the data from `aicodinggym mle download ventilator-pressure-prediction`.
  No original Kaggle data, mirrors, or datasets/models derived from it.
- Do **not** search for, open, or recall-on-request existing solutions to this competition
  (notebooks, discussions, write-ups, repos). Do not web-search this competition at all.
- Ideas must come from our own data analysis and validation errors. If you propose something
  because you remember it from this specific competition rather than from evidence in this repo,
  **say so explicitly** in the chat and in `EXPERIMENTS.md` under "Disclosures" before using it.
- Never put credentials, the gym user ID, or unrelated material in this session: the transcript is
  submitted for grading.

## Fixed evaluation (never change)

- `split.json` — the held-out validation breaths (10% of breaths, stratified by R×C, seed 42).
  6,791 val / 61,114 train breaths. `prep.py` reuses it if it exists. Do not regenerate or edit it.
- Metric: `common.insp_mae` (MAE where `u_out == 0`). Every trial reports it on this split.
- Every trial appends one row to `results.tsv` via `common.log_result` (name, val MAE, snapped
  val MAE, runtime, device, epochs, params, config, note).

## Commands

```bash
python prep.py                                   # once per machine: builds data/cache.npz from data/*.csv
python baseline_gbm.py                           # trial 0 baseline (CPU)
python train.py --name <trial> --epochs N [...]  # sequence model; resumes from checkpoints/<trial>.pt
python train.py --name full_s0 --epochs N --full # fit on train+val for a final submission
python make_submission.py <trial> [<trial> ...]  # median-blend test preds, snap to grid, validate, write submission.csv
aicodinggym mle submit ventilator-pressure-prediction -F submission.csv
```

Outputs: `checkpoints/<trial>.pt`, `outputs/<trial>_val.npy`, `outputs/<trial>_test.npy`,
`outputs/<trial>.out` (stdout log). These are git-ignored; `results.tsv`, `EXPERIMENTS.md`,
`split.json` and code are tracked.

## Working on DataHub

- One GTX 1080 Ti (11 GB), 8 CPUs, 16 GB RAM. The container dies after 6 h (12 h max).
- Run every training in the background with a log, e.g.
  `nohup python -u train.py --name X --epochs 150 > outputs/X.out 2>&1 &`, then poll the log.
  `train.py` checkpoints every 5 epochs and resumes if rerun with the same `--name` and args.
- One training job at a time on the GPU. Check `nvidia-smi` before launching.
- Do not submit to the leaderboard without the user's go-ahead; local validation comes first.

## Loop protocol

1. Read `results.tsv` + `EXPERIMENTS.md`. State one hypothesis and the result that would confirm
   or kill it.
2. Change one thing. New trial name. Train. Record the row.
3. Write 2–3 lines in `EXPERIMENTS.md`: what was tested, MAE and runtime vs. the current best,
   what was learned, what that suggests next. Failed trials are recorded too.
4. If it beats the current best by more than seed noise (~0.003), it becomes the new baseline.
5. Stop and report when the stopping rule in `GOAL.md` is met.

# Goal prompt for the agent in the DataHub container

Paste the block below as your first message (or after `/goal` if your Claude Code version has it).
Edit the numbers if you want a different budget.

```
/goal Reach validation inspiratory MAE <= 0.150 on the fixed split in split.json, then produce
submission.csv from the best model(s).

Context: read CLAUDE.md and EXPERIMENTS.md first. First run `nvidia-smi`, then `python prep.py`
if data/cache.npz is missing, then reproduce the current best trial's config for 5 epochs to
confirm the pipeline works on this GPU and to measure seconds/epoch.

Constraints:
- Follow every rule in CLAUDE.md (data source, no existing solutions, disclose any
  competition-specific idea you did not derive from our own analysis).
- Never change split.json or the metric. One change per trial. Log every trial in results.tsv and
  EXPERIMENTS.md, including failures.
- Budget: at most 5 GPU-hours total and at most 90 minutes per trial. Run training in the
  background with logs and checkpoints; one job at a time.
- Do not submit to AI Coding Gym yourself; tell me when a submission.csv is worth submitting.

Stopping rule — stop at whichever comes first:
(a) val MAE <= 0.150, (b) 5 GPU-hours used, (c) three consecutive trials fail to improve the best
val MAE by more than 0.003.

When you stop, show: the results.tsv table, per-(R,C) val MAE of the best model, the exact
command that produced it, the make_submission.py output for submission.csv, and one paragraph on
what you would try next and why.
```

Smaller goals you can swap in (same constraints):

- "Find why R=50 breaths have ~1.5x the error of the rest: break val error down by R, C and time
  step for the best model, propose one targeted change, and test it."
- "Test whether a 3-seed median blend beats the single best model by more than 0.005 on val."

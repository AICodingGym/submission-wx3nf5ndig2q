"""Shared helpers: the one metric and the one results log every experiment uses."""
import csv
import os
import time

import numpy as np

RESULTS = "results.tsv"
FIELDS = ["time", "name", "val_mae", "val_mae_snapped", "runtime_min", "device", "epochs", "params", "config", "note"]


def load():
    return np.load("data/cache.npz", allow_pickle=False)


def insp_mae(pred, y, u_out):
    """Competition metric: MAE over inspiratory rows (u_out == 0) only."""
    m = u_out == 0
    return float(np.abs(pred[m] - y[m]).mean())


def snap(pred, grid):
    """Round predictions to the nearest value of the discrete pressure grid seen in train."""
    i = np.clip(np.searchsorted(grid, pred), 1, len(grid) - 1)
    lo, hi = grid[i - 1], grid[i]
    return np.where(pred - lo < hi - pred, lo, hi)


def per_setting(pred, y, u_out, rc):
    out = {}
    for s in np.unique(rc):
        k = rc == s
        out[str(s)] = round(insp_mae(pred[k], y[k], u_out[k]), 4)
    return out


def log_result(**row):
    row.setdefault("time", time.strftime("%Y-%m-%d %H:%M"))
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS, delimiter="\t", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)

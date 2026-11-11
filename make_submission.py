"""Blend one or more runs' test predictions into submission.csv and validate the format.

Usage: python make_submission.py run_a run_b ...   [--mean] [--no-snap]
If every named run has outputs/<run>_val.npy, the blend's validation MAE is printed too.
"""
import os
import sys

import numpy as np
import pandas as pd

from common import insp_mae, load, snap

args = [x for x in sys.argv[1:] if not x.startswith("--")]
agg = np.mean if "--mean" in sys.argv else np.median
do_snap = "--no-snap" not in sys.argv
d = load()

if all(os.path.exists(f"outputs/{n}_val.npy") for n in args):
    v = agg(np.stack([np.load(f"outputs/{n}_val.npy") for n in args]), 0)
    yv, uv = d["y"][d["is_val"]], d["u_out"][d["is_val"]]
    print(f"blend val MAE {insp_mae(v, yv, uv):.5f}  snapped {insp_mae(snap(v, d['grid']), yv, uv):.5f}")

t = agg(np.stack([np.load(f"outputs/{n}_test.npy") for n in args]), 0).astype(np.float64)
if do_snap:
    t = snap(t, d["grid"])
sub = pd.DataFrame({"id": d["test_id"], "pressure": t.reshape(-1)})

ss = pd.read_csv("data/sample_submission.csv")
assert list(sub.columns) == list(ss.columns) == ["id", "pressure"]
assert len(sub) == len(ss) and (sub["id"].to_numpy() == ss["id"].to_numpy()).all()
assert np.isfinite(sub["pressure"]).all()
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv: {len(sub)} rows from {args}, pressure range "
      f"[{sub.pressure.min():.2f}, {sub.pressure.max():.2f}], mean {sub.pressure.mean():.3f}")

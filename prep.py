"""Build features + the fixed validation split once, cache as .npz.

Usage: python prep.py            (reads data/train.csv, data/test.csv)
Writes: data/cache.npz, split.json (the held-out breath_ids; never change this file)
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 42
VAL_FRAC = 0.10
DATA = "data"

CONT = [
    "u_in", "u_in_log", "time_step", "dt", "step", "area", "u_in_cumsum",
    "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_lead1", "u_in_lead2", "u_in_lead3",
    "u_in_diff1", "u_in_diff2", "area_over_c", "u_in_x_r", "u_in_cummean",
]
CAT = ["u_out"] + [f"R_{r}" for r in (5, 20, 50)] + [f"C_{c}" for c in (10, 20, 50)] + [
    f"RC_{r}_{c}" for r in (5, 20, 50) for c in (10, 20, 50)
]
FEATS = CONT + CAT


def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("id").reset_index(drop=True)
    g = df.groupby("breath_id", sort=False)
    df["step"] = g.cumcount() / 79.0
    df["dt"] = g["time_step"].diff().fillna(0.0)
    df["u_in_log"] = np.log1p(df["u_in"])
    # Volume delivered is the time-integral of flow: a single-compartment lung has
    # P ~ R*flow + V/C, so cumulative u_in is the physically motivated state variable.
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / (g.cumcount() + 1)
    for k in (1, 2, 3):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0.0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0.0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["area_over_c"] = df["area"] / df["C"]
    df["u_in_x_r"] = df["u_in"] * df["R"]
    for r in (5, 20, 50):
        df[f"R_{r}"] = (df["R"] == r).astype(np.float32)
    for c in (10, 20, 50):
        df[f"C_{c}"] = (df["C"] == c).astype(np.float32)
    for r in (5, 20, 50):
        for c in (10, 20, 50):
            df[f"RC_{r}_{c}"] = ((df["R"] == r) & (df["C"] == c)).astype(np.float32)
    return df


def to_seq(df, cols):
    return df[cols].to_numpy(np.float32).reshape(-1, 80, len(cols))


def main():
    tr = features(pd.read_csv(f"{DATA}/train.csv"))
    te = features(pd.read_csv(f"{DATA}/test.csv"))
    assert (tr.groupby("breath_id").size() == 80).all() and (te.groupby("breath_id").size() == 80).all()

    # Fixed split: hold out whole breaths, stratified by lung setting (R, C).
    b = tr.groupby("breath_id", sort=False)[["R", "C"]].first().reset_index()
    if os.path.exists("split.json"):
        val_ids = set(json.load(open("split.json"))["val_breath_ids"])
    else:
        rng = np.random.default_rng(SEED)
        val_ids = set()
        for _, grp in b.groupby(["R", "C"]):
            ids = np.sort(grp["breath_id"].to_numpy())
            val_ids |= set(rng.choice(ids, size=int(round(len(ids) * VAL_FRAC)), replace=False).tolist())
        json.dump({"seed": SEED, "val_frac": VAL_FRAC, "val_breath_ids": sorted(int(i) for i in val_ids)},
                  open("split.json", "w"))
    is_val = b["breath_id"].isin(val_ids).to_numpy()

    X = to_seq(tr, FEATS)
    Xte = to_seq(te, FEATS)
    nc = len(CONT)
    mu = X[~is_val][:, :, :nc].reshape(-1, nc).mean(0)
    sd = X[~is_val][:, :, :nc].reshape(-1, nc).std(0) + 1e-6
    X[:, :, :nc] = (X[:, :, :nc] - mu) / sd
    Xte[:, :, :nc] = (Xte[:, :, :nc] - mu) / sd

    np.savez(
        f"{DATA}/cache.npz",
        X=X, y=tr["pressure"].to_numpy(np.float32).reshape(-1, 80),
        u_out=tr["u_out"].to_numpy(np.int8).reshape(-1, 80),
        rc=(b["R"].astype(str) + "_" + b["C"].astype(str)).to_numpy().astype("U"),
        is_val=is_val, Xte=Xte, test_id=te["id"].to_numpy(),
        test_u_out=te["u_out"].to_numpy(np.int8).reshape(-1, 80),
        grid=np.sort(tr["pressure"].unique()).astype(np.float64),
    )
    print(f"train breaths {(~is_val).sum()}  val breaths {is_val.sum()}  test breaths {len(Xte)}  feats {len(FEATS)}")


if __name__ == "__main__":
    main()

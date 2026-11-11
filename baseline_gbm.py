"""Trial 0 baseline: per-row gradient boosting on engineered features (no sequence model).

Usage: python baseline_gbm.py
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from common import insp_mae, load, log_result, per_setting, snap

t0 = time.time()
d = load()
X, y, u_out, is_val = d["X"], d["y"], d["u_out"], d["is_val"]
F = X.shape[-1]

mtr = (u_out[~is_val] == 0).reshape(-1)  # fit only on the rows that are scored
model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.1,
                                      max_leaf_nodes=255, random_state=0)
model.fit(X[~is_val].reshape(-1, F)[mtr], y[~is_val].reshape(-1)[mtr])

pv = model.predict(X[is_val].reshape(-1, F)).reshape(-1, 80)
yv, uv = y[is_val], u_out[is_val]
mae, mae_s = insp_mae(pv, yv, uv), insp_mae(snap(pv, d["grid"]), yv, uv)
rt = (time.time() - t0) / 60
print(f"val MAE {mae:.4f}  snapped {mae_s:.4f}  runtime {rt:.1f} min")
print(per_setting(pv, yv, uv, d["rc"][is_val]))
np.save("outputs/gbm_val.npy", pv)
log_result(name="gbm_baseline", val_mae=round(mae, 5), val_mae_snapped=round(mae_s, 5), runtime_min=round(rt, 1),
           device="cpu", epochs="400 iters", params="-", config="HistGB L1, per-row, 34 feats",
           note="baseline: no sequence context beyond lag/lead/cumsum features")

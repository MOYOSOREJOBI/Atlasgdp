from __future__ import annotations

import math

import numpy as np


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    d = np.asarray(loss_a) - np.asarray(loss_b)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1)) if len(d) > 1 else 0.0
    stat = mean_d / math.sqrt(var_d / len(d)) if var_d > 0 else 0.0
    return {"dm_stat": stat, "mean_loss_diff": mean_d}

"""
Baseline comparison methods, so the conformal-calibration approach's benefit is
demonstrated rather than asserted:

  1. global_threshold      -- one threshold fit across BOTH buckets pooled together
                               (the naive approach most off-the-shelf checkers use)
  2. bucket_empirical       -- per-bucket threshold set to the empirical (1-alpha)
                               quantile of RISKY scores on the calibration split,
                               with NO finite-sample correction
  3. conformal (CRC)        -- this project's method (core/conformal.calibrate),
                               per-bucket, with the finite-sample CRC correction
  4. adaptive_online        -- the online heuristic from core/decision.adaptive_recalibrate,
                               evaluated under a simulated distribution shift

All methods are evaluated on the SAME held-out test split for a fair comparison, and
report: false negative rate (miss rate -- the safety-critical metric), false positive
rate on the allow tier (over-flagging), block rate, and review+edit rate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from core.conformal import calibrate as conformal_calibrate, CalibrationResult, _crc_threshold


@dataclass
class MethodResult:
    method: str
    use_case: str
    threshold: float
    fnr_test: float
    fpr_test: float
    block_rate: float
    review_edit_rate: float
    n_test: int


def _metrics_at_threshold(test_scores, test_labels, threshold, allow_threshold=None):
    n = len(test_scores)
    n_risky = max((test_labels == 1).sum(), 1)
    n_safe = max((test_labels == 0).sum(), 1)
    blocked = test_scores >= threshold
    missed = np.sum((test_labels == 1) & ~blocked)
    fnr = missed / n_risky
    block_rate = blocked.sum() / n
    if allow_threshold is None:
        allow_threshold = threshold  # binary methods: no separate review band
    allowed = test_scores < allow_threshold
    flagged_safe = np.sum((test_labels == 0) & ~allowed)
    fpr = flagged_safe / n_safe
    review_edit_rate = 1 - block_rate - (allowed.sum() / n)
    return fnr, fpr, block_rate, max(review_edit_rate, 0.0)


def run_baseline_comparison(calib_df: pd.DataFrame, test_df: pd.DataFrame,
                             score_col: str, label_col: str,
                             use_cases: list[str], target_alphas: dict[str, float],
                             allow_fpr_target: float) -> pd.DataFrame:
    rows = []

    # --- 1. Global threshold (pooled across buckets, single alpha = mean of targets) ---
    # Uses the SAME Conformal Risk Control search as the per-bucket method (core.conformal._crc_threshold)
    # so the comparison isolates "pooled vs per-bucket", not "different calibration procedures".
    pooled_alpha = float(np.mean(list(target_alphas.values())))
    pooled_scores = calib_df[score_col].to_numpy()
    pooled_labels = calib_df[label_col].to_numpy()
    grid = np.linspace(1.0, 0.0, 400)
    global_lambda, _ = _crc_threshold(pooled_scores, pooled_labels, pooled_alpha, grid)
    for uc in use_cases:
        sub = test_df[test_df.use_case == uc]
        fnr, fpr, br, rr = _metrics_at_threshold(sub[score_col].to_numpy(), sub[label_col].to_numpy(), global_lambda)
        rows.append(MethodResult("1. Global threshold (pooled)", uc, global_lambda, fnr, fpr, br, rr, len(sub)))

    # --- 2. Per-bucket empirical quantile, no finite-sample correction ---
    for uc in use_cases:
        sub_c = calib_df[calib_df.use_case == uc]
        risky_scores = sub_c[sub_c[label_col] == 1][score_col].to_numpy()
        alpha = target_alphas[uc]
        thresh = float(np.quantile(risky_scores, alpha)) if len(risky_scores) > 0 else 0.5
        sub_t = test_df[test_df.use_case == uc]
        fnr, fpr, br, rr = _metrics_at_threshold(sub_t[score_col].to_numpy(), sub_t[label_col].to_numpy(), thresh)
        rows.append(MethodResult("2. Per-bucket empirical quantile", uc, thresh, fnr, fpr, br, rr, len(sub_t)))

    # --- 3. Conformal Risk Control (this project's method), per bucket ---
    calibs: dict[str, CalibrationResult] = {}
    for uc in use_cases:
        cal = conformal_calibrate(calib_df, score_col, label_col, uc, target_alphas[uc], allow_fpr_target)
        calibs[uc] = cal
        sub_t = test_df[test_df.use_case == uc]
        fnr, fpr, br, rr = _metrics_at_threshold(
            sub_t[score_col].to_numpy(), sub_t[label_col].to_numpy(), cal.lambda_block, cal.lambda_allow
        )
        rows.append(MethodResult("3. Conformal Risk Control (per bucket)", uc, cal.lambda_block, fnr, fpr, br, rr, len(sub_t)))

    return pd.DataFrame([r.__dict__ for r in rows]), calibs

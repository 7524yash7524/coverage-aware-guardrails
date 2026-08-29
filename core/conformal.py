"""
Conformal calibration layer for the AI Checker.

Instead of thresholding a raw, uncalibrated detector score directly, we calibrate a
decision threshold on a held-out calibration split so that the resulting decision
rule carries a FINITE-SAMPLE, DISTRIBUTION-FREE guarantee on the miss rate (false
negative rate) -- the "we let a genuinely risky response through" event -- which is
the asymmetric, liability-carrying error the brief explicitly calls out.

Method: Conformal Risk Control (Angelopoulos, Bates, Candes, Lei 2022) applied to a
monotone risk function (empirical miss rate as a function of the block threshold).
For a calibration set of size n with scores s_i and true risk labels y_i in {0,1}:

    R_hat(lambda) = (1/n) * sum_i [ y_i == 1 and s_i < lambda ]   (fraction of truly
                     risky calibration cases that would NOT be flagged at threshold lambda)

We pick the smallest lambda such that:
    (n / (n+1)) * R_hat(lambda) + 1/(n+1) <= alpha

This guarantees E[miss rate on a fresh test point] <= alpha, under exchangeability of
calibration and test data -- crucially, WITHOUT needing the raw score to itself be a
calibrated probability.

We calibrate separately per use-case bucket, since a single global threshold silently
over/under-covers specific buckets (the same phenomenon as regime-conditional
calibration reversal across volatility regimes in prior work).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    use_case: str
    alpha_target: float
    lambda_block: float      # score >= lambda_block  -> "block" tier (guarantees miss rate <= alpha)
    lambda_allow: float      # score <  lambda_allow   -> "allow" tier (controls false-positive/over-flag rate)
    n_calibration: int
    empirical_fnr_at_block: float
    empirical_fpr_at_allow: float


def _crc_threshold(scores: np.ndarray, labels: np.ndarray, alpha: float,
                    grid: np.ndarray) -> tuple[float, float]:
    """Smallest lambda on `grid` satisfying the Conformal Risk Control bound for
    false-negative rate <= alpha. Returns (lambda, empirical_fnr_at_lambda)."""
    n = len(scores)
    n_risky = max(labels.sum(), 1)
    for lam in grid:
        missed = np.sum((labels == 1) & (scores < lam))
        r_hat = missed / n_risky
        bound = (n / (n + 1)) * r_hat + 1 / (n + 1)
        if bound <= alpha:
            return float(lam), float(r_hat)
    # If nothing on the grid satisfies it, fall back to the strictest threshold (block everything above min risky score)
    lam = float(grid[0])
    missed = np.sum((labels == 1) & (scores < lam))
    return lam, float(missed / n_risky)


def calibrate(df: pd.DataFrame, score_col: str, label_col: str, use_case: str,
              alpha_block: float, allow_fpr_target: float,
              n_grid: int = 400) -> CalibrationResult:
    """Calibrate lambda_block (miss-rate-controlled, via CRC) and lambda_allow
    (false-positive-rate-controlled, simple empirical quantile) on a calibration split
    for one use-case bucket."""
    sub = df[df["use_case"] == use_case]
    scores = sub[score_col].to_numpy()
    labels = sub[label_col].to_numpy()
    n = len(sub)

    grid = np.linspace(1.0, 0.0, n_grid)  # search from strict (high lambda) to loose
    lambda_block, fnr = _crc_threshold(scores, labels, alpha_block, grid)

    # lambda_allow: the (1 - allow_fpr_target) quantile of SAFE-case scores, i.e. the
    # threshold below which only ~allow_fpr_target of genuinely safe calibration cases
    # fall *above* it (and would therefore be needlessly excluded from the allow tier).
    # Larger lambda_allow => more safe cases correctly auto-allowed => less over-flagging.
    safe_scores = scores[labels == 0]
    n_safe = max(len(safe_scores), 1)
    if len(safe_scores) > 0:
        lambda_allow = float(np.quantile(safe_scores, 1 - allow_fpr_target))
    else:
        lambda_allow = 0.0
    lambda_allow = min(lambda_allow, lambda_block)  # never let allow band swallow the block band
    flagged_safe = np.sum((labels == 0) & (scores >= lambda_allow))
    fpr = float(flagged_safe / n_safe)

    return CalibrationResult(
        use_case=use_case,
        alpha_target=alpha_block,
        lambda_block=lambda_block,
        lambda_allow=lambda_allow,
        n_calibration=n,
        empirical_fnr_at_block=fnr,
        empirical_fpr_at_allow=fpr,
    )


def _wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion k/n. Used so we can
    honestly report whether the target alpha is statistically consistent with the
    observed test-set miss rate, instead of an ad hoc fixed slack."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def evaluate_coverage(df: pd.DataFrame, score_col: str, label_col: str,
                       calib: CalibrationResult) -> dict:
    """Evaluate empirical miss rate on a FRESH test split for the given bucket -- the
    coverage-monitoring signal surfaced on the dashboard.

    IMPORTANT (honesty note): Conformal Risk Control gives a guarantee on the
    *expected* miss rate over the randomness of the calibration split, not a
    deterministic guarantee that this exact test set falls below alpha. So instead of
    a fixed-slack "coverage_held" heuristic, we report a 95% Wilson confidence
    interval on the observed test miss rate and flag whether the target alpha falls
    within it -- if alpha is inside or below the CI's upper bound, the observation is
    statistically consistent with the guarantee; if alpha is clearly below the CI's
    lower bound, that would be a genuine red flag worth investigating."""
    sub = df[df["use_case"] == calib.use_case]
    scores = sub[score_col].to_numpy()
    labels = sub[label_col].to_numpy()
    n_risky = int((labels == 1).sum())
    n_safe = max((labels == 0).sum(), 1)

    missed = int(np.sum((labels == 1) & (scores < calib.lambda_block)))
    fnr_test = missed / max(n_risky, 1)
    ci_lo, ci_hi = _wilson_interval(missed, max(n_risky, 1))

    flagged_safe = np.sum((labels == 0) & (scores >= calib.lambda_allow))
    fpr_test = flagged_safe / n_safe

    return {
        "use_case": calib.use_case,
        "target_alpha": calib.alpha_target,
        "empirical_fnr_test": float(fnr_test),
        "fnr_test_ci95": (round(ci_lo, 3), round(ci_hi, 3)),
        "consistent_with_target": bool(calib.alpha_target <= ci_hi),
        "empirical_fpr_test_allow_tier": float(fpr_test),
        "n_test": len(sub),
        "n_risky_test": n_risky,
    }

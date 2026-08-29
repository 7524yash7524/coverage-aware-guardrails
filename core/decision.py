"""
Tiered decision logic: turns a raw detector score + a bucket's calibrated
(lambda_allow, lambda_block) thresholds into an actionable decision.

    score <  lambda_allow                 -> ALLOW           (in-band, low risk)
    lambda_allow <= score < lambda_block   -> EDIT / REVIEW    (borderline -- uncertain band)
    score >= lambda_block                  -> BLOCK           (high risk, or the CRC bound
                                                                 says letting it through would
                                                                 breach the target miss rate)

The "borderline" band is split into EDIT (closer to allow, cheap auto-redact/rewrite
candidate) vs REVIEW (closer to block, routed to a human) by the midpoint of the band --
this is a simple, explainable rule; a real system would tune this split against reviewer
capacity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.conformal import CalibrationResult


def decide(score: float, calib: CalibrationResult) -> str:
    if score < calib.lambda_allow:
        return "allow"
    if score >= calib.lambda_block:
        return "block"
    midpoint = (calib.lambda_allow + calib.lambda_block) / 2
    return "edit" if score < midpoint else "review"


def apply_decisions(df: pd.DataFrame, score_col: str,
                     calibs: dict[str, CalibrationResult]) -> pd.DataFrame:
    out = df.copy()
    out["decision"] = out.apply(
        lambda r: decide(r[score_col], calibs[r["use_case"]]), axis=1
    )
    return out


# ---- Adaptive conformal recalibration (feedback loop) -------------------------------
def adaptive_recalibrate(working_alpha: float, target_alpha: float, miss_occurred: bool,
                          gamma: float = 0.02, alpha_min: float = 0.01,
                          alpha_max: float = 0.5) -> float:
    """One step of Adaptive Conformal Inference (Gibbs & Candes 2021): after observing
    whether a fresh, reviewed case was a miss (a risky case that was NOT blocked),
    tighten the *working* alpha (used to re-derive lambda_block) if we missed one,
    loosen it slightly if we didn't -- driving the online miss rate back toward the
    fixed long-run target_alpha over time even under distribution drift (a model
    update, a new attack pattern, a seasonal shift). This directly reuses the
    adaptive-CP mechanism from the author's prior Chronos-2 regime-shift calibration
    work."""
    if miss_occurred:
        working_alpha = working_alpha - gamma * (1 - target_alpha)
    else:
        working_alpha = working_alpha + gamma * target_alpha
    return float(np.clip(working_alpha, alpha_min, alpha_max))

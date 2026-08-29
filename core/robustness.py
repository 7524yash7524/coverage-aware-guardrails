"""
Repeated-simulation robustness check: re-runs the full simulate -> split ->
conformal-calibrate -> evaluate pipeline across many random seeds, so we can report
mean/std of the empirical test miss rate per bucket rather than a single-seed number.
This is what turns "it worked on my simulation" into "the method consistently behaves
this way across repeated draws."
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data.simulate import simulate_dataset, BUCKETS
from core.conformal import calibrate, evaluate_coverage


def run_robustness_check(target_alphas: dict[str, float], allow_fpr_target: float,
                          score_col: str = "raw_ensemble_score",
                          n_seeds: int = 60) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        df = simulate_dataset(seed=seed)
        rng = np.random.default_rng(seed)
        is_calib = df.groupby("use_case")["response_id"].transform(
            lambda s: pd.Series(rng.random(len(s)) < 0.5, index=s.index)
        )
        calib_df, test_df = df[is_calib].reset_index(drop=True), df[~is_calib].reset_index(drop=True)

        for uc in BUCKETS:
            cal = calibrate(calib_df, score_col, "true_risk", uc, target_alphas[uc], allow_fpr_target)
            cov = evaluate_coverage(test_df, score_col, "true_risk", cal)
            rows.append({
                "seed": seed,
                "use_case": uc,
                "target_alpha": target_alphas[uc],
                "empirical_fnr_test": cov["empirical_fnr_test"],
            })
    return pd.DataFrame(rows)


def summarize_robustness(raw: pd.DataFrame) -> pd.DataFrame:
    summary = raw.groupby(["use_case", "target_alpha"])["empirical_fnr_test"].agg(
        mean_fnr="mean", std_fnr="std", p5="min", p95="max", n_seeds="count"
    ).reset_index()
    return summary

"""
Simulates AI responses across two enterprise use-case buckets:
  - "customer_facing": real-time, strict risk tolerance (low target alpha)
  - "internal_copilot": batch/internal, looser risk tolerance (higher target alpha)

Each simulated response has a TRUE latent risk label (is it actually a hallucination
/ PII leak / biased response?) plus a set of NOISY, UNCALIBRATED detector scores that
try to estimate that risk. The noise/miscalibration is deliberately different per
bucket and per risk regime, to demonstrate why a single global threshold fails and
why per-bucket conformal calibration is needed.

This mirrors the regime-conditional calibration-reversal finding from the author's
prior Chronos-2 conformal-prediction work: the SAME raw score means different things
in different regimes/buckets.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

BUCKETS = {
    "customer_facing": dict(n=1500, base_risk_rate=0.08, noise_std=0.22, judge_bias=0.05),
    "internal_copilot": dict(n=1500, base_risk_rate=0.15, noise_std=0.12, judge_bias=-0.03),
}


def _simulate_bucket(name: str, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg["n"]
    true_risk = rng.binomial(1, cfg["base_risk_rate"], size=n)  # 1 = genuinely risky response

    # Latent "true" severity for risky cases (0.5-1.0), near-zero for safe cases
    true_severity = np.where(
        true_risk == 1,
        rng.uniform(0.45, 1.0, size=n),
        rng.uniform(0.0, 0.15, size=n),
    )

    # Raw, UNCALIBRATED judge score: true severity + bucket-specific noise + bucket-specific bias.
    # This is what a naive "LLM-as-judge confidence" pipeline would threshold directly.
    raw_judge_score = np.clip(
        true_severity + cfg["judge_bias"] + rng.normal(0, cfg["noise_std"], size=n),
        0, 1,
    )

    # A second, independent-ish detector stream (e.g. retrieval-verification / PII regex score)
    raw_secondary_score = np.clip(
        true_severity * 0.8 + rng.normal(0, cfg["noise_std"] * 1.3, size=n),
        0, 1,
    )

    df = pd.DataFrame({
        "use_case": name,
        "true_risk": true_risk,
        "true_severity": true_severity,
        "raw_judge_score": raw_judge_score,
        "raw_secondary_score": raw_secondary_score,
    })
    # simple ensemble raw score a naive system would threshold on directly
    df["raw_ensemble_score"] = 0.6 * df["raw_judge_score"] + 0.4 * df["raw_secondary_score"]
    return df


def simulate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [_simulate_bucket(name, cfg, rng) for name, cfg in BUCKETS.items()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["response_id"] = [f"resp_{i:05d}" for i in range(len(df))]
    return df


if __name__ == "__main__":
    d = simulate_dataset()
    print(d.groupby("use_case")["true_risk"].mean())
    d.to_csv("data/simulated_responses.csv", index=False)
    print(f"Wrote {len(d)} rows to data/simulated_responses.csv")

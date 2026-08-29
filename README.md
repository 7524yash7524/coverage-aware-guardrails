# Coverage-Aware Guardrails
### A statistically calibrated Responsible AI Checker — Accenture Innovation Challenge 2026, Round 2, Track: ControlPlane.ai

**🔗 Live demo:** [coverage-aware-guardrails.streamlit.app](https://coverage-aware-guardrails-dwuy4twdkxehhtjrvbkvgx.streamlit.app/)

## What this is

Most enterprise AI guardrails threshold a raw, uncalibrated "confidence" score from a
secondary judge model. That score has no guaranteed relationship to true risk, and a
single global threshold silently over-flags some use cases while under-flagging others.

This prototype replaces the naive threshold with a **conformal calibration layer**: a
decision threshold calibrated per use-case bucket so the resulting allow/edit/review/block
decision carries a **finite-sample, distribution-free guarantee** on the miss rate (letting
a genuinely risky response through) — verified empirically on held-out data, not asserted.

It builds directly on prior published work calibrating time-series foundation models
([SSRN 7300198](https://doi.org/10.2139/ssrn.7300198)), where the same phenomenon —
a single global calibration silently failing across regimes — motivated per-regime
conformal calibration. Here, "regime" becomes "use case" (customer-facing vs. internal).

## Architecture

```
AI Response → Detection Layer (parallel scorers) → Conformal Calibration Layer
            → Tiered Decision (allow/edit/review/block) → Feedback Loop (adaptive CP)
```

See `business_proposal.md` for the full write-up (problem framing, business case,
roadmap, risks & mitigations).

## What's implemented in this prototype

- `data/simulate.py` — simulates AI responses across two use-case buckets
  (`customer_facing`, `internal_copilot`) with injected ground-truth risk labels and
  deliberately different, bucket-specific noise/bias in the raw detector scores —
  demonstrating why a global threshold fails.
- `core/conformal.py` — **Conformal Risk Control** (Angelopoulos, Bates, Candès, Lei,
  2022) calibration of the block threshold to guarantee miss rate ≤ target α, plus a
  quantile-based allow threshold to control over-flagging of safe responses. Calibrated
  separately per bucket. Coverage is reported with a 95% Wilson confidence interval on
  the test-set miss rate, not a fixed-slack pass/fail — CRC's guarantee is on the
  *expected* miss rate over calibration-split randomness, and the CI makes that honest.
- `core/decision.py` — tiered decision logic (allow / edit / review / block) from the
  calibrated thresholds, plus an **online adaptive heuristic** (Adaptive-Conformal-Inference-
  style) for threshold movement under drift — explicitly labeled as a heuristic in the
  dashboard, since its trailing-window quantile re-derivation does not carry the same
  finite-sample guarantee as the CRC calibration in Section 2.
- `core/baselines.py` — baseline comparison: a pooled global threshold vs. an uncorrected
  per-bucket empirical quantile vs. this project's per-bucket CRC, evaluated on the same
  held-out split, so the value of per-bucket + finite-sample-corrected calibration is
  demonstrated rather than asserted.
- `core/robustness.py` — repeats the full simulate → calibrate → evaluate pipeline across
  many seeds and reports mean/std/range of the test miss rate per bucket, so results
  aren't reported from a single lucky draw.
- `app.py` — Streamlit dashboard: (1) shows why a global threshold fails, (2) shows
  calibrated thresholds and verifies the coverage guarantee (with CI) on held-out data,
  (3) shows the decision-tier breakdown, (4) simulates a distribution shift mid-stream
  and shows the adaptive heuristic pulling the miss rate back toward target, (5) a
  baseline comparison table, (6) a repeated-seed robustness check, (7) an audit-trail
  sample.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Adjust the per-use-case α (risk tolerance) and the over-flag tolerance live in the
sidebar — this is the governance/policy layer described in the proposal.

## Minimum expectations addressed

- Detection techniques: heuristic scoring streams simulating judge + retrieval-verification
  detectors (Section 1 of `app.py`).
- Decision logic: confidence scoring → calibrated risk bands → tiered response (Section 3).
- Governance: per-use-case, sidebar-configurable policy (α, over-flag tolerance).
- Feedback loop: adaptive online heuristic under simulated drift (Section 4), clearly
  scoped as a heuristic rather than a formal re-derivation of the CRC guarantee.
- Metrics & monitoring: empirical miss rate vs. target with a 95% CI, over-flag rate,
  baseline comparison, and repeated-seed robustness (Sections 2, 5, 6).

## What's simplified / next steps

This is a Phase 0 prototype on simulated data with a single ensemble detector score.
See `business_proposal.md` §5 (Phased Roadmap) for how this extends to real bias/PII
detector streams, a fully re-derived CRC threshold inside the adaptive loop, multi-turn/
agentic compounding risk, and a real labeled pilot dataset.

## Author

Yash Chitransh

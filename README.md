# Coverage-Aware Guardrails

A calibrated Responsible AI checker. Built for the Accenture Innovation Challenge 2026, Round 2 (ControlPlane.ai track).

**Live demo:** [coverage-aware-guardrails.streamlit.app](https://coverage-aware-guardrails-dwuy4twdkxehhtjrvbkvgx.streamlit.app/) &nbsp;|&nbsp; **Business proposal:** [`business_proposal.md`](business_proposal.md) &nbsp;|&nbsp; **Built on:** [SSRN 7300198](https://doi.org/10.2139/ssrn.7300198)

## The problem

Most AI guardrails block or allow a response by thresholding a raw "confidence" score from a judge model. That score has no guaranteed link to real risk, and one global threshold over-flags some use cases while under-flagging others.

## The idea

Calibrate the decision threshold **separately for each use case** with Conformal Risk Control (Angelopoulos, Bates, Candes, Lei, 2022). This gives a bound on the **expected miss rate**: how often a truly risky response gets through. It needs no assumption that the raw score is a real probability. The same idea comes from my earlier paper, where one global calibration failed across volatility regimes. Here "regime" becomes "use case" (customer-facing vs. internal).

```
AI response -> detector scores -> conformal calibration -> allow / edit / review / block -> adaptive feedback loop
```

## Results (simulated data, dashboard defaults)

Setup: 3,000 simulated responses (1,500 per use case), random 50/50 calibration/test split, seed 42, ensemble detector score. Risk rates are 8% (customer-facing) and 15% (internal copilot). Targets: miss rate at most 5% and 15%.

**One run (seed 42), miss rate on the test split:**

| Method | Customer-facing (target 5%) | Internal copilot (target 15%) |
|---|---:|---:|
| Global threshold (pooled) | 7.0% | 11.3% |
| Per-bucket empirical quantile | 7.0% | 13.0% |
| Per-bucket Conformal Risk Control | 7.0% | 13.0% |

On this single run the customer-facing miss rate (7.0%) is above its 5% target, and all three methods land close to each other. The guarantee is about the *average*, so I also repeated the whole pipeline across seeds.

**60 repeated runs, test miss rate:**

| Use case | Target | Mean | Std | Min to max |
|---|---:|---:|---:|---|
| Customer-facing | 5% | 5.05% | 3.85 pts | 0.0% to 15.2% |
| Internal copilot | 15% | 16.05% | 4.88 pts | 4.5% to 26.4% |

The average tracks the target, but a single run can be far off. Each test bucket has few truly risky cases, so the miss rate is noisy. This is why the dashboard reports a confidence interval and a multi-seed check instead of a single number.

**What conformal calibration adds on this data:** a controlled threshold per use case and a "review / edit" band between allow and block (1.1% and 10.8% of responses in the two buckets on seed 42). It did not clearly beat the simpler per-bucket baseline on miss rate in the single run.

## What is in the repo

| File | Purpose |
|---|---|
| `data/simulate.py` | Simulates responses for two use-case buckets with ground-truth risk labels and bucket-specific detector noise |
| `core/conformal.py` | Conformal Risk Control threshold per bucket, plus an allow threshold that limits over-flagging; 95% Wilson interval on the test miss rate |
| `core/decision.py` | Tiered decisions (allow / edit / review / block) and an online adaptive heuristic for drift. The heuristic is labelled as such because it does not carry the finite-sample guarantee |
| `core/baselines.py` | Global threshold vs. per-bucket quantile vs. per-bucket CRC, on the same test split |
| `core/robustness.py` | Repeats simulate, calibrate and evaluate across many seeds |
| `app.py` | Streamlit dashboard with seven sections: why a global threshold fails, coverage check with interval, decision tiers, drift simulation, baseline table, robustness check, audit trail |

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Risk tolerance per use case and the over-flag tolerance are sliders in the sidebar. This is the "governance policy" layer from the proposal.

## Limits and next steps

- Phase 0 prototype on **simulated** data with a single ensemble detector score. Real detectors and a real labelled pilot set are the next step.
- The guarantee is on expected miss rate under exchangeable data. It does not cover a single run or a drifting stream.
- The adaptive loop is a heuristic. Re-deriving the CRC threshold inside the loop is listed in the roadmap in `business_proposal.md` (section 5).

## Author

Yash Chitransh, M.Sc. Mathematics & Scientific Computing, NIT Warangal. [LinkedIn](https://www.linkedin.com/in/yash-chitransh-4b108b2aa)

## License

MIT, see `LICENSE`.

# Coverage-Aware Guardrails: A Statistically Calibrated Responsible AI Checker
### Accenture Innovation Challenge 2026 — Round 2 — Track: ControlPlane.ai
**Team:** Yash Chitransh

---

## 1. Problem Framing

Enterprises now run generative AI across many simultaneous use cases — customer-facing chatbots, internal
copilots, decision-support tools in regulated workflows — each with a different risk signature, latency budget,
and regulatory exposure. Round 1 proposed a Responsible AI Checker that flags bias, hallucination, and privacy
risk before a response reaches a user. The hard problem in Round 2 is not detection alone — every enterprise AI
vendor already ships an "AI-as-judge" detector. The hard problem is **decision logic**: turning a noisy, uncalibrated
risk score into an actionable, trustworthy allow/edit/review/block decision, tuned differently per use case, without
drowning reviewers in false positives or silently missing real harms.

Most checkers on the market solve this with an arbitrary threshold on an LLM judge's raw confidence score.
That confidence score is **not a calibrated probability** — an LLM saying "I am 90% confident this is not a
hallucination" has no guaranteed relationship to the true 90% event frequency. This is the same problem that
exists in any black-box forecasting system, and it is exactly the problem **conformal prediction** was built to solve:
turning arbitrary model scores into **statistically guaranteed** coverage intervals, regardless of the underlying
model's own (mis)calibration.

## 2. Solution Design

**Core idea:** Wrap the AI Checker's detection layer in a conformal calibration layer, so that every flag/block
decision comes with a distribution-free, empirically verifiable coverage guarantee — "when we say a response's
risk is below threshold X, that holds true (1-α) of the time, provably, on held-out data" — rather than an
opaque confidence number.

### 2.1 Architecture

```
 ┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐     ┌──────────────┐
 │  AI Response │ --> │  Detection Layer │ --> │ Conformal          │ --> │ Tiered       │
 │ (LLM output) │     │ (parallel scorers)│     │ Calibration Layer │     │ Decision     │
 └─────────────┘     └──────────────────┘     └───────────────────┘     └──────────────┘
                              |                          |                       |
                     hallucination score          per-use-case,          allow / edit /
                     bias score                   per-α calibrated       flag-for-review /
                     PII/entity score              risk bands            block
                                                          |
                                                  ┌───────────────┐
                                                  │ Feedback Loop │  <- overridden/labeled
                                                  │ (adaptive CP) │     cases recalibrate
                                                  └───────────────┘     thresholds online
```

- **Detection layer:** runs in parallel to protect latency. Includes: (a) a lightweight NER/regex PII detector,
  (b) an embedding-similarity retrieval-verification score against source documents (for RAG-style hallucination
  detection), (c) a secondary LLM-as-judge score for bias/tone. Each detector emits a raw, uncalibrated score.
- **Conformal calibration layer (the novel core):** each raw score stream is calibrated using split conformal
  prediction (static use cases) or **adaptive conformal prediction / Venn-Abers predictors** (use cases with
  distribution drift over time — e.g., a customer support assistant whose query mix shifts seasonally). Calibration
  is performed **per use-case bucket**, because — exactly as found in prior work calibrating time-series
  foundation models across volatility regimes — a single global calibration silently over- or under-covers
  specific sub-populations. Customer-facing / real-time buckets get a stricter α (lower miscoverage tolerance);
  internal / batch buckets can tolerate a looser α in exchange for lower reviewer load.
- **Tiered decision logic:** the calibrated risk band (not the raw score) drives the action: allow (in-band, low
  risk), edit (borderline, auto-redact/rewrite), flag-for-review (uncertain, human-in-the-loop), block (high risk
  or compounding risk in a multi-turn/agentic session).
- **Feedback loop:** every reviewed, overridden, or human-labeled case is fed back into the calibration set,
  and thresholds are recalibrated using an online/adaptive conformal update — so calibration drift (e.g., a model
  update, a new attack pattern, a seasonal shift in queries) is detected and corrected rather than silently eroding
  trust.
- **Governance layer:** a configuration file per use case defines α (target miscoverage), which detectors are
  active, latency budget, and escalation rules — giving compliance teams an auditable, versioned policy surface
  rather than buried thresholds in code.

### 2.2 Why this beats a naive threshold-on-confidence approach
- **Provable, not asserted:** coverage guarantees hold under minimal assumptions (exchangeability), independent
  of whether the underlying LLM judge is well-calibrated.
- **Tunable, auditable tradeoff:** the over-flagging/under-flagging tension the brief calls out becomes an explicit,
  reportable α per use case, not an ad hoc knob.
- **Detects its own drift:** empirical coverage is continuously measured against target coverage; a widening gap
  is itself the monitoring signal for when detectors, not just thresholds, need retraining.

## 3. Target Users
- **Enterprise AI/ML platform teams** deploying multiple LLM-backed products who need one governance layer
  across all of them.
- **Compliance/Risk officers** who need an auditable, quantifiable answer to "how sure are you this is safe?"
  rather than a vibes-based confidence score.
- **Human reviewers** (support ops, content moderation) who need a manageable flag queue, not alert fatigue.

## 4. Business Case & Impact
- Tens of thousands of interactions/week across use cases (per brief's reference parameters) means even a
  few-percent reduction in false-positive review load translates into meaningful reviewer-hours saved, while
  the coverage guarantee reduces liability exposure from missed harms — the two costs the brief explicitly frames
  as asymmetric and irreducible without deliberate tuning.
- A single governance layer that generalizes across use cases and geographies (via per-bucket α and policy
  config) avoids the enterprise anti-pattern of building a bespoke checker per product.

## 5. Phased Roadmap
- **Phase 0 (this prototype):** single detection stream (hallucination/retrieval-verification), split-conformal
  calibration, 2 simulated use-case buckets, static replay dataset, coverage/FP-FN dashboard.
- **Phase 1:** add bias and PII detectors as parallel streams; adaptive conformal recalibration on synthetic drift.
- **Phase 2:** multi-turn/agentic compounding-risk tracking; integrate human-override capture into a live
  feedback loop.
- **Phase 3:** pilot on a real internal use case with a small labeled review set; measure actual reviewer load
  reduction and empirical coverage against target.

## 6. Key Risks & Mitigations
| Risk | Mitigation |
|---|---|
| No reliable real-time ground truth to check claims against | Use retrieval-verification against source docs where available; where not, calibrate against a held-out human-labeled sample rather than requiring ground truth at inference time — conformal prediction only needs exchangeable calibration data, not per-query ground truth. |
| Exchangeability assumption breaks under real distribution drift (model updates, new attack types) | Adaptive conformal prediction continuously re-estimates coverage and widens/narrows intervals in response — this is the same mechanism used to handle regime shifts in the underlying time-series calibration work this proposal builds on. |
| Latency budget for real-time use cases | Calibration lookup is O(1) (precomputed quantile table per bucket); only the underlying detector scoring, not calibration, adds latency — calibration is nearly free. |
| Regulatory rules evolve and vary by geography | Policy/α is externalized to config, not hard-coded, so a new jurisdiction's tolerance is a config change, not a redeploy. |

## 7. Prototype Scope (for this submission)
A limited but working demonstration of the core mechanism: simulated LLM responses with injected
hallucination/PII cases → detection scores → split conformal calibration into risk bands per use-case bucket →
tiered decision output → coverage metrics with confidence intervals, benchmarked against two baselines (a
naive pooled global threshold, and an uncorrected per-bucket empirical quantile) and validated for
consistency across 40+ repeated simulation seeds.

## 8. Statistical Honesty Notes
Two claims in this prototype are deliberately scoped precisely rather than overstated:
- **Coverage is a distributional guarantee, not a per-sample certainty.** Conformal Risk Control
  guarantees the *expected* miss rate over the randomness of the calibration split is ≤ target α — a
  single test draw fluctuates around that target. The dashboard reports a 95% confidence interval on
  the observed miss rate rather than a fixed pass/fail slack.
- **The online drift-adaptation mechanism (§4 of the dashboard) is an adaptive heuristic, not a formal
  re-derivation of the Conformal Risk Control guarantee** — it demonstrates the direction and mechanism
  of online threshold movement under distribution shift, using a cheap trailing-window quantile rather
  than a full CRC recomputation at each step. A production system would re-run full CRC on a rolling
  calibration window instead.

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data.simulate import simulate_dataset, BUCKETS
from core.conformal import calibrate, evaluate_coverage
from core.decision import apply_decisions, adaptive_recalibrate
from core.baselines import run_baseline_comparison
from core.robustness import run_robustness_check, summarize_robustness

st.set_page_config(page_title="Coverage-Aware Guardrails", layout="wide")

st.title("Coverage-Aware Guardrails")
st.caption(
    "A conformally-calibrated Responsible AI Checker — ControlPlane.ai, Round 2 prototype. "
    "Raw detector scores are turned into decisions with a *provable* miss-rate guarantee, "
    "calibrated separately per use case."
)

# --------------------------------------------------------------------------------------
# Governance config (sidebar) -- the policy layer
# --------------------------------------------------------------------------------------
st.sidebar.header("Governance policy config")
st.sidebar.write("Per-use-case risk tolerance (α = target max miss rate on truly risky responses).")
alpha_customer = st.sidebar.slider("customer_facing target α", 0.01, 0.30, 0.05, 0.01)
alpha_internal = st.sidebar.slider("internal_copilot target α", 0.01, 0.30, 0.15, 0.01)
allow_fpr_target = st.sidebar.slider("Max tolerated over-flag rate on safe responses (allow tier)", 0.01, 0.30, 0.10, 0.01)
score_choice = st.sidebar.selectbox("Detector score used", ["raw_ensemble_score", "raw_judge_score", "raw_secondary_score"])
seed = st.sidebar.number_input("Simulation seed", value=42, step=1)

ALPHAS = {"customer_facing": alpha_customer, "internal_copilot": alpha_internal}

# --------------------------------------------------------------------------------------
# Simulate + split calibration/test
# --------------------------------------------------------------------------------------
df = simulate_dataset(seed=int(seed))
rng = np.random.default_rng(int(seed))
is_calib = df.groupby("use_case")["response_id"].transform(
    lambda s: pd.Series(rng.random(len(s)) < 0.5, index=s.index)
)
calib_df, test_df = df[is_calib].reset_index(drop=True), df[~is_calib].reset_index(drop=True)

calibs = {
    uc: calibrate(calib_df, score_choice, "true_risk", uc, ALPHAS[uc], allow_fpr_target)
    for uc in BUCKETS
}

test_df = apply_decisions(test_df, score_choice, calibs)

# --------------------------------------------------------------------------------------
# Section 1: why global thresholds fail
# --------------------------------------------------------------------------------------
st.header("1. Why a single global threshold fails")
c1, c2 = st.columns(2)
for col, uc in zip((c1, c2), BUCKETS):
    sub = calib_df[calib_df["use_case"] == uc]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sub[sub.true_risk == 0][score_choice], name="safe (true_risk=0)",
                                opacity=0.6, nbinsx=30, marker_color="#2ca02c"))
    fig.add_trace(go.Histogram(x=sub[sub.true_risk == 1][score_choice], name="risky (true_risk=1)",
                                opacity=0.6, nbinsx=30, marker_color="#d62728"))
    fig.update_layout(barmode="overlay", title=f"{uc}: raw score distribution", height=320,
                       margin=dict(t=40, b=10))
    col.plotly_chart(fig, width='stretch')
st.caption(
    "The same raw score means something different in each bucket — a fixed global threshold "
    "would over-cover one bucket and under-cover the other. Calibration is done separately per bucket."
)

# --------------------------------------------------------------------------------------
# Section 2: calibrated thresholds + coverage guarantee
# --------------------------------------------------------------------------------------
st.header("2. Calibrated thresholds and coverage guarantee (held-out test split)")
rows = []
for uc, cal in calibs.items():
    cov = evaluate_coverage(test_df, score_choice, "true_risk", cal)
    lo, hi = cov["fnr_test_ci95"]
    rows.append({
        "use_case": uc,
        "target α (max miss rate)": cal.alpha_target,
        "λ_allow": round(cal.lambda_allow, 3),
        "λ_block": round(cal.lambda_block, 3),
        "empirical miss rate (test)": round(cov["empirical_fnr_test"], 3),
        "95% CI (Wilson)": f"[{lo}, {hi}]",
        "consistent with target?": "✅" if cov["consistent_with_target"] else "⚠️ investigate",
        "over-flag rate on safe (allow tier, test)": round(cov["empirical_fpr_test_allow_tier"], 3),
        "n calibration": cal.n_calibration,
        "n risky (test)": cov["n_risky_test"],
    })
st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
st.caption(
    "λ_block is chosen via Conformal Risk Control (Angelopoulos et al. 2022), which guarantees the "
    "*expected* miss rate over the randomness of the calibration split is ≤ target α. A single test "
    "draw will fluctuate around that target, so rather than a fixed pass/fail slack, we report a 95% "
    "Wilson interval on the observed test miss rate: if the target α falls within (or below) that "
    "interval, the observation is statistically consistent with the guarantee."
)

# --------------------------------------------------------------------------------------
# Section 3: decision tier breakdown
# --------------------------------------------------------------------------------------
st.header("3. Tiered decisions on held-out traffic")
tier_counts = test_df.groupby(["use_case", "decision"]).size().reset_index(name="count")
fig2 = go.Figure()
for uc in BUCKETS:
    sub = tier_counts[tier_counts.use_case == uc]
    fig2.add_trace(go.Bar(name=uc, x=sub["decision"], y=sub["count"]))
fig2.update_layout(barmode="group", height=350, title="Decisions by tier and use case",
                    margin=dict(t=40, b=10))
st.plotly_chart(fig2, width='stretch')

missed = test_df[(test_df.true_risk == 1) & (test_df.decision.isin(["allow", "edit"]))]
st.metric("Genuinely risky responses that slipped past allow/edit (should be rare, bounded by α)",
          len(missed))

# --------------------------------------------------------------------------------------
# Section 4: adaptive recalibration under drift (feedback loop)
# --------------------------------------------------------------------------------------
st.header("4. Feedback loop: adaptive *heuristic* recalibration under drift")
st.warning(
    "⚠️ **Honesty note:** this section is an **online heuristic**, not a re-derivation of the formal "
    "Conformal Risk Control guarantee from Section 2. It adapts the working α (Adaptive Conformal "
    "Inference-style, Gibbs & Candès 2021) and re-derives λ_block as a **trailing-window empirical "
    "quantile** for speed — that quantile step does *not* carry the finite-sample CRC correction, so "
    "no formal coverage guarantee is claimed here. It demonstrates the *direction and mechanism* of "
    "online adaptation under drift; a production version would re-run full CRC on a rolling calibration "
    "window instead of a plain quantile."
)
st.write(
    "Simulating a stream of reviewed cases for `customer_facing` with a sudden distribution shift "
    "halfway through (e.g. a model update). The working α tightens automatically after misses and "
    "gradually relaxes otherwise, pulling the realized miss rate back toward target without a manual "
    "re-tune."
)
n_stream = 400
shift_at = n_stream // 2
target_a = alpha_customer
working_alpha = target_a
sub = calib_df[calib_df.use_case == "customer_facing"].sample(n=n_stream, replace=True, random_state=1).reset_index(drop=True)
# inject a post-shift bias to simulate drift
drift = np.where(np.arange(n_stream) >= shift_at, 0.15, 0.0)
stream_scores = np.clip(sub[score_choice] + drift, 0, 1)

history = []
lam = calibs["customer_facing"].lambda_block
for i in range(n_stream):
    s, y = stream_scores[i], sub.true_risk[i]
    miss = (y == 1) and (s < lam)
    working_alpha = adaptive_recalibrate(working_alpha, target_a, miss)
    # re-derive lambda_block from the *current* working alpha via a simple empirical quantile
    # over a trailing window (cheap online approximation to full CRC recompute)
    window = stream_scores[max(0, i - 60):i + 1]
    lam = float(np.quantile(window, working_alpha)) if len(window) > 5 else lam
    history.append({"i": i, "working_alpha": working_alpha, "lambda_block": lam, "miss": miss})

hist_df = pd.DataFrame(history)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=hist_df.i, y=hist_df.working_alpha, name="working α", line=dict(color="#1f77b4")))
fig3.add_vline(x=shift_at, line_dash="dash", line_color="red", annotation_text="distribution shift")
fig3.update_layout(height=320, title="Working α over time (adapts after the shift)", margin=dict(t=40, b=10))
st.plotly_chart(fig3, width='stretch')
st.caption(f"Misses before shift: {hist_df.iloc[:shift_at].miss.sum()} / {shift_at} · "
           f"Misses after shift: {hist_df.iloc[shift_at:].miss.sum()} / {n_stream - shift_at} "
           "(the adaptive loop pulls this back down after an initial spike).")

with st.expander("Governance / audit trail (per-decision, sample)"):
    st.dataframe(test_df[["response_id", "use_case", score_choice, "true_risk", "decision"]].head(25),
                 width='stretch', hide_index=True)

# --------------------------------------------------------------------------------------
# Section 5: baseline comparison
# --------------------------------------------------------------------------------------
st.header("5. Baseline comparison — does conformal calibration actually help?")
st.write(
    "Same held-out test split, three methods: a single global threshold pooled across both use cases "
    "(the naive approach), a per-bucket threshold with no finite-sample correction, and this project's "
    "per-bucket Conformal Risk Control. All three are evaluated on identical data for a fair comparison."
)
with st.spinner("Running baseline comparison..."):
    bl_df, _ = run_baseline_comparison(calib_df, test_df, score_choice, "true_risk",
                                        list(BUCKETS.keys()), ALPHAS, allow_fpr_target)
display_bl = bl_df.copy()
display_bl["target_alpha"] = display_bl["use_case"].map(ALPHAS)
display_bl = display_bl[["method", "use_case", "target_alpha", "threshold", "fnr_test", "fpr_test",
                          "block_rate", "review_edit_rate"]]
display_bl.columns = ["method", "use_case", "target α", "threshold", "FNR (test)", "FPR (allow tier)",
                       "block rate", "review/edit rate"]
st.dataframe(display_bl.round(3), width='stretch', hide_index=True)
st.caption(
    "Look for: (a) does the global threshold's FNR drift further from target α in one bucket than the "
    "other (the miscalibration-across-regimes problem)? (b) does the uncorrected per-bucket quantile "
    "run noticeably hotter than its target α on this finite sample, while CRC stays controlled?"
)

# --------------------------------------------------------------------------------------
# Section 6: robustness across repeated simulations
# --------------------------------------------------------------------------------------
st.header("6. Robustness across repeated simulations")
st.write(
    "A single seed can look good by chance. Re-running the full simulate → calibrate → evaluate "
    "pipeline across many seeds shows whether the conformal method's test miss rate consistently "
    "tracks target α, or whether Section 2's result was a lucky draw."
)
n_seeds = st.slider("Number of seeds", 10, 100, 40, 10)
if st.button("Run robustness check"):
    with st.spinner(f"Running {n_seeds} independent simulations..."):
        raw = run_robustness_check(ALPHAS, allow_fpr_target, score_choice, n_seeds=n_seeds)
        summary = summarize_robustness(raw)
    st.dataframe(summary.round(4), width='stretch', hide_index=True)
    fig4 = go.Figure()
    for uc in BUCKETS:
        sub = raw[raw.use_case == uc]
        fig4.add_trace(go.Box(y=sub.empirical_fnr_test, name=uc))
        fig4.add_hline(y=ALPHAS[uc], line_dash="dash",
                        annotation_text=f"{uc} target α={ALPHAS[uc]}")
    fig4.update_layout(height=380, title="Distribution of test-set miss rate across seeds",
                        margin=dict(t=40, b=10))
    st.plotly_chart(fig4, width='stretch')
else:
    st.caption("Click the button to run — kept manual since it re-runs the full pipeline N times.")

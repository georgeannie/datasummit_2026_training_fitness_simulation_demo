# pages/1_Scenario_Lab.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.nav import SCENARIO_PAGE, BELIEFS_PAGE, render_top_nav

from core import (
    get_history_and_beliefs,
    simulate_uncertainty,
    recommend_plan,
)

st.set_page_config(page_title="Scenario Lab — 10K Training", layout="wide")
render_top_nav(active=SCENARIO_PAGE)
st.title("Scenario Lab — 10K Training Allocation Under Uncertainty")

st.markdown(
    "**Decision:** Given limited time, how should I allocate training next week to improve performance while managing injury risk?"
)

df_daily, df_proxy, beliefs = get_history_and_beliefs(seed=11, weeks=6)

# Use last completed week as context
wk_context = df_proxy.iloc[-1].to_dict()
baseline_soreness = beliefs.baseline_soreness_14d

# Helpful context signals (story-friendly)
weeks_observed = int(df_proxy.shape[0])
missed_total = int(df_proxy["missed_days"].sum())
missed_last = int(wk_context["missed_days"])
avg_sleep_last = float(wk_context["avg_sleep"])
avg_soreness_last = float(wk_context["avg_soreness"])
baseline_soreness_14d = float(beliefs.baseline_soreness_14d)

# Simple narrative statuses (no math on screen)
def recovery_label(soreness_last, sleep_last):
    if soreness_last >= 5.2 and sleep_last <= 6.6:
        return "Strained"
    if soreness_last >= 5.2:
        return "Sore"
    if sleep_last <= 6.6:
        return "Under-recovered"
    return "Stable"

def consistency_label(missed_total):
    # rough but intuitive
    if missed_total <= 1:
        return "Strong"
    if missed_total <= 3:
        return "Mixed"
    return "Fragile"

recovery_state = recovery_label(avg_soreness_last, avg_sleep_last)
consistency_state = consistency_label(missed_total)

# =========================
# TOP ROW: Context KPI cards
# =========================
st.subheader("What we know so far")
k1, k2, k3, k4 = st.columns(4, gap="large")

k1.metric("Consistency (last 6 weeks)", consistency_state, f"{missed_total} missed days", border=True)
k2.metric("Recovery trend (last week)", recovery_state, f"Soreness {avg_soreness_last:.1f} | Sleep {avg_sleep_last:.1f}h", border=True)
k3.metric("Last week disruptions", f"{missed_last} missed", "This affects uncertainty", border=True, delta_color="off")
k4.metric("Baseline soreness (14d)", f"{baseline_soreness_14d:.1f}",
        "Used for guardrails", border=True, delta_color="off", 
        help="""**What this represents** 
Baseline soreness is a rolling measure of accumulated training stress. This baseline represents your **sustainable operating level**.

It reflects:
- Recent training load (harder weeks increase it)
- Recovery quality (sleep reduces it)
- Carryover from previous days

When soreness stays above baseline for multiple days:
- Stress is accumulating faster than recovery
- Risk compounds even if performance looks good
"""
)

# =========================
# TOP ROW: Sliders (move controls up)
# =========================
st.subheader("Set next week's constraints")
s1, s2, s3, s4 = st.columns([1.2, 1.3, 1.3, 1.2], gap="large")

with s1:
    total_min = st.slider(
        "Total time (minutes)",
        min_value=120,
        max_value=480,
        value=300,
        step=10
    )
    st.caption("Capacity is fixed. Decisions are allocations.")

with s2:
    easy_min = st.slider("Easy minutes", 0, total_min, int(0.60 * total_min), 10)

with s3:
    remaining = total_min - easy_min
    tempo_min = st.slider("Moderate/Tempo minutes", 0, remaining, int(0.30 * total_min), 10)
    strength_min = total_min - easy_min - tempo_min
    st.caption(f"Strength minutes (auto): **{strength_min}**")

with s4:
    risk_posture = st.slider(
        "Risk posture",
        0.0, 1.0, 0.35, 0.05,
        help="0 = conservative, 1 = aggressive"
    )
    st.caption("Controls tolerance for downside outcomes.")

st.markdown("---")
#c1, c2, c3 = st.columns([1.1, 1.4, 1.5], gap="large")

# with c1:
#     st.subheader("Inputs")

#     total_min = st.slider(
#         "Total training time next week (minutes)",
#         min_value=120, max_value=480, value=300, step=10
#     )
#     st.caption("Capacity is fixed. Optimization happens within constraints.")

#     st.markdown("### Allocate minutes")
#     easy_min = st.slider("Easy", 0, total_min, int(0.60 * total_min), 10)
#     remaining = total_min - easy_min
#     tempo_min = st.slider("Tempo", 0, remaining, int(0.30 * total_min), 10)
#     strength_min = total_min - easy_min - tempo_min
#     st.write(f"**Strength:** {strength_min} min (auto)")

#     risk_posture = st.slider("Risk posture (Conservative → Aggressive)", 0.0, 1.0, 0.35, 0.05)
#     st.caption("Controls tolerance for downside outcomes.")

#     st.markdown("---")
#     st.markdown("### Context (last 6 weeks)")
#     st.write(f"- Weeks observed: **{df_proxy.shape[0]}**")
#     st.write(f"- Last-week avg soreness: **{wk_context['avg_soreness']:.2f}**")
#     st.write(f"- Last-week avg sleep: **{wk_context['avg_sleep']:.2f} hrs**")
#     st.write(f"- Baseline soreness (14d): **{baseline_soreness:.2f}**")
c1, c2 = st.columns([2, 2], gap="large")

with c1:
    st.subheader("Expected Impact (with Uncertainty)")

    imps, risks = simulate_uncertainty(easy_min, tempo_min, strength_min, wk_context, beliefs, n=2500, seed=999)
    p10, p50, p90 = np.percentile(imps, [10, 50, 90])
    e_risk = float(np.mean(risks))

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Expected Improvement (P50)", f"{p50:,.0f}")
    kpi2.metric("Downside (P10)", f"{p10:,.0f}")
    kpi3.metric("Injury Risk (avg)", f"{100*e_risk:,.1f}%")

    hist_y, hist_x = np.histogram(imps, bins=35)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=hist_x[:-1], y=hist_y, name="Simulated outcomes"))
    for val, name in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        fig_dist.add_vline(x=float(val), line_width=2, line_dash="dash", annotation_text=name)

    fig_dist.update_layout(
        title="Expected Performance Impact (Next 2–3 Weeks)",
        xaxis_title="Improvement score (unitless)",
        yaxis_title="Simulation count",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.caption(
        "Interpretation: This is a range of plausible outcomes—not a single prediction. "
        "The width of the distribution is the uncertainty."
    )

with c2:
    st.subheader("Trade-offs & Recommendation")

    # Frontier: sample feasible allocations quickly
    rng = np.random.default_rng(202)
    points = []
    for _ in range(40):
        tempo = int(rng.integers(low=0, high=int(0.45 * total_min) + 1))
        strength = int(rng.integers(low=int(0.05 * total_min), high=int(0.25 * total_min) + 1))
        easy = total_min - tempo - strength
        if easy < 0:
            continue

        imps_s, risks_s = simulate_uncertainty(
            easy, tempo, strength, wk_context, beliefs, n=700, seed=int(rng.integers(1, 1_000_000))
        )
        points.append(
            dict(easy=easy, tempo=tempo, strength=strength,
                 exp_imp=float(np.mean(imps_s)),
                 exp_risk=float(np.mean(risks_s)))
        )

    frontier = pd.DataFrame(points)

    rec = recommend_plan(total_min, wk_context, beliefs, risk_posture)

    # Current selection point
    e_imp = float(np.mean(imps))
    e_risk = float(np.mean(risks))

    fig_frontier = go.Figure()
    fig_frontier.add_trace(go.Scatter(
        x=frontier["exp_imp"], y=frontier["exp_risk"] * 100,
        mode="markers", name="Feasible plans",
        text=[f"Easy {r.easy} | Tempo {r.tempo} | Strength {r.strength}" for r in frontier.itertuples(index=False)],
        hovertemplate="%{text}<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
    ))
    fig_frontier.add_trace(go.Scatter(
        x=[e_imp], y=[e_risk*100],
        mode="markers", name="Your plan",
        marker=dict(size=14, symbol="diamond"),
        hovertemplate="Your plan<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
    ))
    fig_frontier.add_trace(go.Scatter(
        x=[rec["e_imp"]], y=[rec["e_risk"]*100],
        mode="markers", name="Recommended",
        marker=dict(size=16, symbol="star"),
        hovertemplate="Recommended<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
    ))

    fig_frontier.update_layout(
        title="Performance vs Injury Risk Trade-off",
        xaxis_title="Expected improvement (mean)",
        yaxis_title="Injury risk proxy (%)",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    st.markdown("### Recommended Training Plan")
    st.markdown(f"**Easy:** {rec['easy']} min  \n**Tempo:** {rec['tempo']} min  \n**Strength:** {rec['strength']} min")

    st.markdown("**Why this plan**")
    st.markdown(
        "- Robust across plausible futures (optimizes risk-adjusted outcome)\n"
        "- Avoids the high-injury tail while keeping strong expected gains"
    )

    st.markdown("**Guardrail**")
    st.markdown(
        f"Reduce **tempo by 20%** if soreness stays above **{baseline_soreness:.1f}** for **2 consecutive days**."
    )

    st.caption("Interpretation: This is not the best plan—it’s the most robust one.")

# Debug only
with st.expander("Debug: daily data"):
    st.dataframe(df_daily, use_container_width=True)
with st.expander("Debug: weekly proxy"):
    st.dataframe(df_proxy, use_container_width=True)





# # scenario_lab.py
# import math
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from datetime import date, timedelta

# # -----------------------------
# # 0) Page config
# # -----------------------------
# st.set_page_config(
#     page_title="Scenario Lab — 10K Training Allocation",
#     layout="wide"
# )

# # -----------------------------
# # 2) Core “beliefs” (used implicitly here)
# #    - Saturating response per lever
# #    - Carryover via simple decay
# #    - Risk threshold via nonlinear load
# # -----------------------------
# def sat_gain(m, alpha, k, h):
#     """Saturation curve: alpha * m^h / (k^h + m^h)"""
#     m = max(0.0, float(m))
#     return alpha * (m ** h) / ((k ** h) + (m ** h) + 1e-9)

# def compute_expected_improvement(easy, tempo, strength, params):
#     """
#     A compact performance score (unitless):
#     - easy contributes steadily
#     - tempo contributes more but saturates faster
#     - strength contributes modestly (stability)
#     Weekly training produces benefits that extend into future weeks — so the
#       “decision impact window” is ~2–3 weeks, not 7 days.

#     1.00x would mean “all benefit is immediate; nothing carries over” (false)
#     >1.00x means “some of the benefit persists” (true)
#     1.25x is a conservative way to say:

#     You get your primary benefit this week
#     Plus about 25% additional benefit realized over the next ~1–2 weeks
#     Conclusion: modeling decision structure: training has lagged effects,
#       and the decision window is longer than one week.”
#     """
#     g_easy = sat_gain(easy, **params["easy"])
#     g_tempo = sat_gain(tempo, **params["tempo"])
#     g_strength = sat_gain(strength, **params["strength"])

#     # simple carryover: assume next 2–3 weeks capture 1.25x of weekly benefit
#     # (we keep it fixed and interpret it narratively; uncertainty is simulated elsewhere)
#     # A more complex model could have lever-specific carryover or nonlinear decay
#     # For simplicity, we apply the same multiplier to the total weekly gain, 
#     # which implies that all training effects have similar carryover patterns.
#     #
#     carryover_multiplier = 1.25
#     return carryover_multiplier * (g_easy + g_tempo + g_strength)


# def compute_injury_risk_proxy(easy, tempo, strength, wk_context):
#     """
#     Risk proxy:
#     - tempo is higher strain (weight 1.6)
#     - soreness baseline increases risk
#     - nonlinear threshold (sigmoid-like)
#     """
#     # weighted load
#     load = easy + 1.6 * tempo + 0.9 * strength

#     # contextual risk: higher if prior soreness high / sleep low
#     soreness = wk_context["avg_soreness"]
#     sleep = wk_context["avg_sleep"]

#     context = 0.10 * max(0, soreness - 4.0) + 0.08 * max(0, 7.0 - sleep)

#     # thresholded nonlinearity around ~320-380 min "equivalent load"
#     x = (load / 360.0) + context
#     risk = 1.0 / (1.0 + math.exp(-8.0 * (x - 1.0)))  # 0..1
#     return float(np.clip(risk, 0.0, 1.0))


# def simulate_uncertainty(easy, tempo, strength, wk_context, n=2000, seed=42):
#     """
#     Monte Carlo simulation:
#     - parameter uncertainty on response curves
#     - weekly shock driven by recovery context
#     Returns:
#       improvement_samples, risk_samples
#     """
#     rng = np.random.default_rng(seed)

#     # base parameters (interpretable, stable)
#     # alpha = max potential gain from that lever (units of improvement score)
#     # k = half-saturation point (minutes to reach ~50% of max gain)
#     # h = curve steepness (higher h = more all-or-nothing)
#     # These are the "best guess" parameters around which we will simulate uncertainty.
#     # The values are chosen to reflect domain knowledge:
#     # - Easy runs have a steady contribution that saturates around 200-300 min.
#     # - Tempo runs have a higher potential gain but saturate faster (around 150-200 min).
#     # - Strength training has a modest contribution that saturates quickly (around 80-120 min). 
#     # The h values reflect the intuition that tempo is more all-or-nothing, while easy runs have a more gradual curve.
#     # The uncertainty simulation will add noise to these parameters to reflect our lack of precise knowledge about the true response curves,
#     #  and how they might vary based on individual differences and contextual factors.
#     # Note: these parameters are not fitted to real data; they are illustrative for the purpose of the demo. In a real application, 
#     # you would want to calibrate these parameters based on empirical data or expert input.
#     # The key point is that the model captures the qualitative relationships and allows us to 
#     # explore how uncertainty in these parameters affects our decisions.

#     base = {
#         "easy": dict(alpha=120.0, k=240.0, h=1.20),
#         "tempo": dict(alpha=170.0, k=170.0, h=1.15),
#         "strength": dict(alpha=60.0, k=90.0, h=1.10),
#     }

#     # uncertainty scales: bigger when data is messy (high soreness / low sleep / missed days)
#     soreness = wk_context["avg_soreness"]
#     sleep = wk_context["avg_sleep"]
#     missed = wk_context["missed_days"]

#     instability = 0.08 + 0.03 * max(0, soreness - 4.0) + 0.02 * max(0, 7.0 - sleep) + 0.02 * missed
#     instability = float(np.clip(instability, 0.08, 0.25))

#     imps = []
#     risks = []

#     for _ in range(n):
#         # randomize alphas/k/h with mild lognormal noise
#         params = {}
#         for key in ["easy", "tempo", "strength"]:
#             b = base[key]
#             params[key] = dict(
#                 alpha=b["alpha"] * rng.lognormal(mean=0.0, sigma=instability),
#                 k=b["k"] * rng.lognormal(mean=0.0, sigma=instability),
#                 h=float(np.clip(b["h"] + rng.normal(0, 0.05), 1.0, 1.4)),
#             )

#         imp = compute_expected_improvement(easy, tempo, strength, params)

#         # add outcome noise (week-to-week variability)
#         outcome_noise = rng.normal(0, 10.0 + 35.0 * instability)
#         imp = max(0.0, imp + outcome_noise)

#         r = compute_injury_risk_proxy(easy, tempo, strength, wk_context)
#         # risk uncertainty mildly correlated with instability
#         r = float(np.clip(r + rng.normal(0, 0.06 * (1 + 3 * instability)), 0.0, 1.0))

#         imps.append(imp)
#         risks.append(r)

#     return np.array(imps), np.array(risks)


# # -----------------------------
# # 3) Recommendation engine (grid search)
# # -----------------------------
# def recommend_plan(total_min, wk_context, risk_posture):
#     """
#     Recommend allocation by maximizing utility:
#       utility = E[improvement] - lambda * E[risk]
#     lambda depends on risk_posture
#     """
#     # convert posture into a penalty (bigger penalty = more conservative)
#     # risk_posture in [0, 1] where 0=conservative, 1=aggressive
#     lam = float(np.interp(risk_posture, [0, 1], [140.0, 40.0]))

#     step = 10  # minutes
#     best = None

#     # keep search small and reliable
#     for easy in range(0, total_min + 1, step):
#         for tempo in range(0, total_min - easy + 1, step):
#             strength = total_min - easy - tempo

#             # sanity constraints (avoid extreme unrealistic plans)
#             if tempo > 0.45 * total_min:
#                 continue
#             if strength < 0.05 * total_min:
#                 continue

#             imps, risks = simulate_uncertainty(easy, tempo, strength, wk_context, n=900, seed=123)
#             e_imp = float(np.mean(imps))
#             e_risk = float(np.mean(risks))
#             utility = e_imp - lam * e_risk

#             if best is None or utility > best["utility"]:
#                 best = dict(
#                     easy=easy,
#                     tempo=tempo,
#                     strength=strength,
#                     e_imp=e_imp,
#                     e_risk=e_risk,
#                     utility=utility,
#                 )

#     return best


# # -----------------------------
# # 4) UI — Scenario Lab
# # -----------------------------
# st.title("Scenario Lab — 10K Training Allocation Under Uncertainty")

# st.markdown(
#     "**Decision:** Given limited time, how should I allocate training next week to improve performance while managing injury risk?"
# )

# df_daily = generate_daily_data(seed=11, weeks=6)
# df_weekly = weekly_aggregate(df_daily)

# # Use last completed week as context
# wk_context_row = df_weekly.iloc[-1].to_dict()

# # Compute baseline soreness from last 2 weeks for guardrail
# baseline_soreness = float(df_daily.tail(14)["soreness"].mean())

# # ---------------- Left column: Inputs ----------------
# c1, c2, c3 = st.columns([1.1, 1.4, 1.5], gap="large")

# with c1:
#     st.subheader("Inputs")

#     total_min = st.slider(
#         "Total training time next week (minutes)",
#         min_value=120,
#         max_value=480,
#         value=300,
#         step=10
#     )

#     st.caption("Capacity is fixed. Optimization happens within constraints.")

#     # Allocation sliders — user-controlled
#     st.markdown("### Allocate minutes")
#     easy_min = st.slider("Easy", 0, total_min, int(0.60 * total_min), 10)
#     remaining = total_min - easy_min
#     tempo_min = st.slider("Tempo", 0, remaining, int(0.30 * total_min), 10)
#     strength_min = total_min - easy_min - tempo_min

#     st.write(f"**Strength:** {strength_min} min (auto)")

#     risk_posture = st.slider(
#         "Risk posture (Conservative → Aggressive)",
#         0.0, 1.0, 0.35, 0.05
#     )
#     st.caption("Controls tolerance for downside outcomes.")

#     st.markdown("---")
#     st.markdown("### Context (last 6 weeks)")
#     st.write(f"- Weeks observed: **{df_weekly.shape[0]}**")
#     st.write(f"- Last-week avg soreness: **{wk_context_row['avg_soreness']:.2f}**")
#     st.write(f"- Last-week avg sleep: **{wk_context_row['avg_sleep']:.2f} hrs**")
#     st.write(f"- Baseline soreness (14d): **{baseline_soreness:.2f}**")


# # ---------------- Middle column: Uncertainty distribution + KPIs ----------------
# with c2:
#     st.subheader("Expected Impact (with Uncertainty)")

#     imps, risks = simulate_uncertainty(
#         easy_min, tempo_min, strength_min, wk_context_row, n=2500, seed=999
#     )

#     p10, p50, p90 = np.percentile(imps, [10, 50, 90])
#     e_imp = float(np.mean(imps))
#     e_risk = float(np.mean(risks))

#     kpi1, kpi2, kpi3 = st.columns(3)
#     kpi1.metric("Expected Improvement (P50)", f"{p50:,.0f}")
#     kpi2.metric("Downside (P10)", f"{p10:,.0f}")
#     kpi3.metric("Injury Risk (avg)", f"{100*e_risk:,.1f}%")

#     # Distribution chart (P10/P50/P90 markers)
#     hist_y, hist_x = np.histogram(imps, bins=35)
#     fig_dist = go.Figure()
#     fig_dist.add_trace(go.Bar(x=hist_x[:-1], y=hist_y, name="Simulated outcomes"))
#     for val, name in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
#         fig_dist.add_vline(x=float(val), line_width=2, line_dash="dash", annotation_text=name)

#     fig_dist.update_layout(
#         title="Expected Performance Impact (Next 2–3 Weeks)",
#         xaxis_title="Improvement score (unitless)",
#         yaxis_title="Simulation count",
#         height=380,
#         margin=dict(l=20, r=20, t=60, b=20)
#     )
#     st.plotly_chart(fig_dist, use_container_width=True)

#     st.caption(
#         "Interpretation: This is a range of plausible outcomes—not a single prediction. "
#         "The width of the distribution is the uncertainty."
#     )


# # ---------------- Right column: Frontier + recommendation ----------------
# with c3:
#     st.subheader("Trade-offs & Recommendation")

#     # Frontier: sample feasible allocations quickly
#     rng = np.random.default_rng(202)
#     points = []
#     for _ in range(40):
#         # random feasible plan
#         tempo = int(rng.integers(low=0, high=int(0.45 * total_min) + 1))
#         strength = int(rng.integers(low=int(0.05 * total_min), high=int(0.25 * total_min) + 1))
#         easy = total_min - tempo - strength
#         if easy < 0:
#             continue

#         imps_s, risks_s = simulate_uncertainty(easy, tempo, strength, wk_context_row, n=700, seed=int(rng.integers(1, 1_000_000)))
#         points.append(
#             dict(easy=easy, tempo=tempo, strength=strength,
#                  exp_imp=float(np.mean(imps_s)),
#                  exp_risk=float(np.mean(risks_s)))
#         )

#     frontier = pd.DataFrame(points)

#     # Recommended plan from grid search
#     rec = recommend_plan(total_min, wk_context_row, risk_posture)

#     fig_frontier = go.Figure()
#     fig_frontier.add_trace(
#         go.Scatter(
#             x=frontier["exp_imp"],
#             y=frontier["exp_risk"] * 100,
#             mode="markers",
#             name="Feasible plans",
#             text=[
#                 f"Easy {r.easy} | Tempo {r.tempo} | Strength {r.strength}"
#                 for r in frontier.itertuples(index=False)
#             ],
#             hovertemplate="%{text}<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
#         )
#     )

#     # highlight current user-selected plan
#     fig_frontier.add_trace(
#         go.Scatter(
#             x=[e_imp],
#             y=[e_risk * 100],
#             mode="markers",
#             name="Your plan",
#             marker=dict(size=14, symbol="diamond"),
#             hovertemplate="Your plan<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
#         )
#     )

#     # highlight recommendation
#     fig_frontier.add_trace(
#         go.Scatter(
#             x=[rec["e_imp"]],
#             y=[rec["e_risk"] * 100],
#             mode="markers",
#             name="Recommended",
#             marker=dict(size=16, symbol="star"),
#             hovertemplate="Recommended<br>Expected: %{x:.0f}<br>Risk: %{y:.1f}%<extra></extra>"
#         )
#     )

#     fig_frontier.update_layout(
#         title="Performance vs Injury Risk Trade-off",
#         xaxis_title="Expected improvement (mean)",
#         yaxis_title="Injury risk proxy (%)",
#         height=380,
#         margin=dict(l=20, r=20, t=60, b=20)
#     )
#     st.plotly_chart(fig_frontier, use_container_width=True)

#     # Decision card (the climax)
#     st.markdown("### Recommended Training Plan")
#     st.markdown(
#         f"""
# **Easy:** {rec['easy']} min  
# **Tempo:** {rec['tempo']} min  
# **Strength:** {rec['strength']} min
# """
#     )

#     st.markdown("**Why this plan**")
#     st.markdown(
#         "- Robust across plausible futures (optimizes risk-adjusted outcome)\n"
#         "- Avoids the high-injury tail while keeping strong expected gains"
#     )

#     # Guardrail based on soreness baseline
#     st.markdown("**Guardrail**")
#     st.markdown(
#         f"Reduce **tempo by 20%** if soreness stays above **{baseline_soreness:.1f}** for **2 consecutive days**."
#     )

#     st.caption("Interpretation: This is not the best plan—it’s the most robust one.")

# # -----------------------------
# # Optional: show generated data (for you, not for the talk)
# # -----------------------------
# with st.expander("Show generated daily data (debug only)"):
#     st.dataframe(df_daily, use_container_width=True)

# with st.expander("Show weekly aggregation (debug only)"):
#     st.dataframe(df_weekly, use_container_width=True)
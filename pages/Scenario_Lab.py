# pages/1_Scenario_Lab.py — Conference-ready version
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.nav import SCENARIO_PAGE, render_top_nav

from core import (
    get_history_and_beliefs,
    simulate_uncertainty,
    weighted_load,
    sigmoid,
    recommend_plan,
)

# ---------------------------
# Conference UI constants
# ---------------------------
AXIS_TITLE_FONT_SIZE = 28
AXIS_TICK_FONT_SIZE = 22
ANNOTATION_FONT_SIZE = 26
PERCENTILE_LINE_WIDTH = 5
CHART_HEIGHT = 480
MARKER_CANDIDATE = 14
MARKER_CURRENT = 22
MARKER_REC = 24

FIXED_STRENGTH_MIN = 60
MIN_EASY_MIN = 30
STEP = 10
SIM_N_MAIN = 2500
SIM_N_FRONTIER = 700
SIM_N_REC = 900

# Colors
C_BLUE = "#2563EB"
C_GREEN = "#059669"
C_RED = "#DC2626"
C_CANDIDATE = "#60A5FA"


# ---------------------------
# Helpers
# ---------------------------
def style_plot_axes(fig):
    fig.update_xaxes(showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                     tickfont=dict(size=AXIS_TICK_FONT_SIZE))
    fig.update_yaxes(showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                     tickfont=dict(size=AXIS_TICK_FONT_SIZE))


def _min_to_hm(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h}h {mm:02d}m" if h else f"{mm}m"


def belief_risk_pct_for_plan(easy, tempo, strength, beliefs):
    plan_load = weighted_load(easy, tempo, strength)
    risk_pct = float(sigmoid(beliefs.risk_slope * (plan_load - beliefs.risk_threshold)) * 100)
    return plan_load, risk_pct


def risk_limit_from_posture(risk_posture):
    return float(np.interp(risk_posture, [0.0, 1.0], [52.0, 62.0]))


# ---------------------------
# Page header
# ---------------------------
render_top_nav(active=SCENARIO_PAGE)

st.markdown("""
<div style="margin-bottom:8px;">
  <div style="font-size:44px; font-weight:800; color:#1A1A1A; line-height:1.1;">
    Scenario Lab
  </div>
  <p style="font-size:24px; color:#6B7280; line-height:1.5; margin-top:8px;">
    10K Training Allocation Under Uncertainty
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Slider label */
div[data-testid="stSlider"] > label p {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { transform: scaleY(1.2); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Data / beliefs
# ---------------------------
df_daily, df_proxy, beliefs = get_history_and_beliefs(seed=11, weeks=6)
wk_context = df_proxy.iloc[-1].to_dict()
baseline_soreness_14d = float(beliefs.baseline_soreness_14d)

missed_total = int(df_proxy["missed_days"].sum())
missed_last = int(wk_context["missed_days"])
avg_sleep_last = float(wk_context["avg_sleep"])
avg_soreness_last = float(wk_context["avg_soreness"])


def recovery_label(soreness_last, sleep_last):
    if soreness_last >= 5.2 and sleep_last <= 6.6: return "Strained"
    if soreness_last >= 5.2: return "Sore"
    if sleep_last <= 6.6: return "Under-recovered"
    return "Stable"


def consistency_label(missed_total):
    if missed_total <= 1: return "Strong"
    if missed_total <= 3: return "Mixed"
    return "Fragile"


recovery_state = recovery_label(avg_soreness_last, avg_sleep_last)
consistency_state = consistency_label(missed_total)

# =========================
# TOP ROW: Context cards
# =========================
st.markdown('<div style="font-size:32px; font-weight:800; color:#1A1A1A; margin-bottom:12px;">What we know so far</div>',
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="large")

top_kpi_cards = [
    ("Consistency (6 weeks)", consistency_state, f"{missed_total} missed days", "#c44747", "#f3dede"),
    ("Recovery (last week)", recovery_state, f"Soreness {avg_soreness_last:.1f} | Sleep {avg_sleep_last:.1f}h", "#c44747", "#f3dede"),
    ("Disruptions (last week)", f"{missed_last} missed", "Affects uncertainty", "#c44747", "#f3dede"),
    ("System strain (14-day avg)", f"{baseline_soreness_14d:.1f}", "Used as guardrail", "#c44747", "#f3dede"),
]

for col, (label, value, detail, detail_color, detail_bg) in zip([k1, k2, k3, k4], top_kpi_cards):
    col.markdown(
        f"""
        <div style="border:2px solid #1622a3; border-radius:12px; padding:18px 18px 16px; min-height:170px;">
            <div style="font-size:18px; font-weight:600; line-height:1.3; margin-bottom:8px;">{label}</div>
            <div style="font-size:42px; font-weight:800; line-height:1.0; margin-bottom:12px;">{value}</div>
            <div style="
                display:inline-block;
                font-size:16px;
                font-weight:600;
                color:{detail_color};
                background:{detail_bg};
                border-radius:999px;
                padding:6px 14px;
                line-height:1.2;
            ">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- init session state ---
if "total_min" not in st.session_state:
    st.session_state.total_min = 400
if "interval_min" not in st.session_state:
    st.session_state.interval_min = 140
if "easy_min" not in st.session_state:
    st.session_state.easy_min = 220
if "last_changed" not in st.session_state:
    st.session_state.last_changed = "interval"


def _reconcile_allocation():
    total = int(st.session_state.total_min)
    remaining = max(0, total - FIXED_STRENGTH_MIN)
    min_easy = min(MIN_EASY_MIN, remaining)

    if remaining < MIN_EASY_MIN:
        total = MIN_EASY_MIN + FIXED_STRENGTH_MIN
        st.session_state.total_min = total
        remaining = total - FIXED_STRENGTH_MIN
        min_easy = MIN_EASY_MIN

    if st.session_state.last_changed == "interval":
        interval = int(st.session_state.interval_min)
        interval = max(0, min(interval, remaining - min_easy))
        easy = remaining - interval
        easy = max(min_easy, min(easy, remaining))
        interval = remaining - easy
    else:
        easy = int(st.session_state.easy_min)
        easy = max(min_easy, min(easy, remaining))
        interval = remaining - easy
        interval = max(0, interval)

    st.session_state.easy_min = int(easy)
    st.session_state.interval_min = int(interval)


def _on_total_change():
    _reconcile_allocation()

def _on_interval_change():
    st.session_state.last_changed = "interval"
    _reconcile_allocation()

def _on_easy_change():
    st.session_state.last_changed = "easy"
    _reconcile_allocation()


# ---- Sliders ----
s1, s2, s3, s4 = st.columns([1.2, 1.3, 1.3, 1.2], gap="large")

with s1:
    st.slider("Total time (minutes)", min_value=120, max_value=480,
              value=int(st.session_state.total_min), step=STEP,
              key="total_min", on_change=_on_total_change)
    total_min = int(st.session_state.total_min)
    remaining = max(0, total_min - FIXED_STRENGTH_MIN)
    st.markdown(f"""
    <div style="font-size:2.2rem; font-weight:500; margin-top:-0.2rem; margin-bottom:0.5rem;">
        {total_min} min <span style="font-size:1.6rem; font-weight:700; opacity:0.75;">({_min_to_hm(total_min)})</span>
    </div>
    <div style="font-size:1.4rem; font-weight:600; opacity:0.65; margin-top:-0.1rem;">
        Strength: {FIXED_STRENGTH_MIN} min (fixed)
    </div>
    """, unsafe_allow_html=True)

with s2:
    min_easy = min(MIN_EASY_MIN, remaining)
    interval_max = max(0, remaining - min_easy)
    st.slider("Interval Training (minutes)", min_value=0, max_value=int(interval_max),
              value=int(min(st.session_state.interval_min, interval_max)),
              step=STEP, key="interval_min", on_change=_on_interval_change)
    interval_min = int(st.session_state.interval_min)
    st.markdown(f"""
    <div style="font-size:2.2rem; font-weight:500; margin-top:-0.2rem;">
        {interval_min} min <span style="font-size:1.6rem; font-weight:700; opacity:0.75;">({_min_to_hm(interval_min)})</span>
    </div>
    """, unsafe_allow_html=True)

with s3:
    min_easy = min(MIN_EASY_MIN, remaining)
    easy_max = remaining
    st.slider("Easy Run (minutes)", min_value=int(min_easy), max_value=int(easy_max),
              value=int(min(max(st.session_state.easy_min, min_easy), easy_max)),
              step=STEP, key="easy_min", on_change=_on_easy_change)
    easy_min = int(st.session_state.easy_min)
    st.markdown(f"""
    <div style="font-size:2.2rem; font-weight:500; margin-top:-0.2rem;">
        {easy_min} min <span style="font-size:1.6rem; font-weight:700; opacity:0.75;">({_min_to_hm(easy_min)})</span>
    </div>
    """, unsafe_allow_html=True)

with s4:
    risk_posture = st.slider("Risk Appetite", 0.0, 1.0, 0.35, 0.05,
                             help="0 = conservative, 1 = aggressive")
    st.markdown(f"""
    <div style="font-size:2.2rem; font-weight:500; margin-top:-0.2rem;">
        {risk_posture:.2f} <span style="font-size:1.6rem; font-weight:700; opacity:0.75;">
        ({'Aggressive' if risk_posture >= 0.5 else 'Conservative'})
        </span>
    </div>
    """, unsafe_allow_html=True)

# Allocation summary
strength_min = FIXED_STRENGTH_MIN
allocated_min = int(st.session_state.easy_min) + int(st.session_state.interval_min) + strength_min

st.markdown(f"""
<div style="font-size:2.0rem; font-weight:800; margin-top:-0.2rem; margin-bottom:0.5rem;">
    Total allocated: {allocated_min} / {st.session_state.total_min} min
    <span style="font-size:1.5rem; opacity:0.75;">
        ({_min_to_hm(allocated_min)} / {_min_to_hm(st.session_state.total_min)})
    </span>
    <span style='color:#059669; margin-left:0.6rem;'>✅</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# Simulation for CURRENT plan
# =========================
imps, risks = simulate_uncertainty(easy_min, interval_min, strength_min, wk_context, beliefs, n=SIM_N_MAIN, seed=999)
p10, p50, p90 = np.percentile(imps, [10, 50, 90])
e_imp = float(np.mean(imps))
e_risk_pct = float(np.mean(risks)) * 100

# =========================
# Main layout (charts)
# =========================
c1, c2 = st.columns([2, 2], gap="large")

with c1:
    st.markdown('<div style="font-size:30px; font-weight:800; margin-bottom:12px;">Expected Impact (with Uncertainty)</div>',
                unsafe_allow_html=True)

    kpi1, kpi2, kpi3 = st.columns(3)
    cards = [
        ("Typical outcome (P50)", f"{p50:,.0f}"),
        ("Chance of breaking down", f"{e_risk_pct:,.1f}%"),
    ]
    for col, (label, value) in zip([kpi1, kpi2, kpi3], cards):
        col.markdown(f"""
        <div style="font-size:18px; font-weight:600; line-height:1.25; margin-bottom:6px;">{label}</div>
        <div style="font-size:36px; font-weight:800; line-height:1.0;">{value}</div>
        """, unsafe_allow_html=True)

    hist_y, hist_x = np.histogram(imps, bins=35)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=hist_x[:-1], y=hist_y, name="Simulated outcomes",
        marker_color=C_BLUE, marker_line_width=0))

    for val, name in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        fig_dist.add_vline(
            x=float(val), line_width=PERCENTILE_LINE_WIDTH, line_dash="dash",
            annotation_text=name, annotation_font_size=ANNOTATION_FONT_SIZE)

    fig_dist.update_layout(
        title=dict(text="Performance Impact (Next 2–3 Weeks)", xanchor="left", x=0.0, font=dict(size=32)),
        xaxis_title="Improvement score",
        yaxis_title="Simulation count",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=CHART_HEIGHT, margin=dict(l=25, r=20, t=70, b=55))
    fig_dist.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig_dist.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    style_plot_axes(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True,
                    config={"displayModeBar": False, "displaylogo": False})

with c2:
    show_tradeoff_chart = st.toggle("Show trade-off chart", value=False)

    if show_tradeoff_chart:
        st.markdown('<div style="font-size:30px; font-weight:800; margin-bottom:12px;">Trade-offs & Recommendation</div>',
                    unsafe_allow_html=True)

        with st.spinner("Building trade-off frontier..."):

            total_min = int(st.session_state.total_min)
            strength_min = FIXED_STRENGTH_MIN
            risk_limit_pct = risk_limit_from_posture(risk_posture)

            remaining = total_min - strength_min
            min_easy = min(MIN_EASY_MIN, remaining)

            frontier_points = []
            for tempo in range(0, remaining - min_easy + 1, STEP):
                easy = remaining - tempo
                imps_s, risks_s = simulate_uncertainty(
                    easy=easy, tempo=tempo, strength=strength_min,
                    wk_context=wk_context, beliefs=beliefs,
                    n=SIM_N_FRONTIER, seed=202 + tempo)

                load = weighted_load(easy, tempo, strength_min)
                belief_risk_pct = float(
                    sigmoid(beliefs.risk_slope * (load - beliefs.risk_threshold)) * 100)

                frontier_points.append({
                    "easy": int(easy), "tempo": int(tempo), "strength": int(strength_min),
                    "exp_imp": float(np.mean(imps_s)),
                    "exp_risk": float(np.mean(risks_s)) * 100.0,
                    "load": float(load), "belief_risk_pct": belief_risk_pct,
                })

            frontier = pd.DataFrame(frontier_points)

            if frontier.empty:
                st.error("No candidate plans could be generated.")
                st.stop()

            rec = recommend_plan(
                total_min=total_min, wk_context=wk_context, beliefs=beliefs,
                risk_posture=risk_posture, risk_limit_pct=risk_limit_pct,
                fixed_strength=FIXED_STRENGTH_MIN, step=STEP,
                n_sims=SIM_N_REC, seed=123, min_easy_min=MIN_EASY_MIN)

            if rec is None:
                st.error("Unable to generate a recommendation.")
                st.stop()

            risk_limit_used_pct = float(rec["risk_limit_used_pct"])

            # ── Trade-off chart ──
            fig_frontier = go.Figure()

            line_label = (
                f"Risk limit = {risk_limit_used_pct:.1f}%"
                if rec["constraint_met"]
                else f"Min achievable risk = {risk_limit_used_pct:.1f}%")

            fig_frontier.add_hline(
                y=risk_limit_used_pct, line_dash="dash",
                line_width=PERCENTILE_LINE_WIDTH,
                line_color="rgba(220,0,0,0.85)",
                annotation_text=line_label,
                annotation_position="top left",
                annotation_font_size=ANNOTATION_FONT_SIZE,
                annotation_font_color="rgba(220,0,0,0.95)")

            # Candidate plans
            fig_frontier.add_trace(go.Scatter(
                x=frontier["exp_imp"], y=frontier["exp_risk"],
                mode="markers", name="Candidate plans",
                customdata=np.stack([frontier["load"], frontier["belief_risk_pct"]], axis=1),
                text=[f"Easy {r.easy} | Interval {r.tempo} | Strength {r.strength}"
                      for r in frontier.itertuples(index=False)],
                hovertemplate=(
                    "%{text}<br>Expected: %{x:.0f}<br>Breakdown risk: %{y:.1f}%"
                    "<br>Load: %{customdata[0]:.0f}<extra></extra>"),
                marker=dict(size=MARKER_CANDIDATE, color=C_CANDIDATE, opacity=0.6,
                            line=dict(width=2, color="rgba(0,0,0,0.5)"))))

            # Current plan
            fig_frontier.add_trace(go.Scatter(
                x=[e_imp], y=[e_risk_pct],
                mode="markers+text", name="Current",
                text=["Current"], textposition="top center",
                textfont=dict(size=MARKER_CURRENT, color=C_GREEN),
                marker=dict(size=MARKER_CURRENT, symbol="diamond", color=C_GREEN),
                hovertemplate="Current<br>Expected: %{x:.0f}<br>Breakdown risk: %{y:.1f}%<extra></extra>"))

            # Recommended plan
            fig_frontier.add_trace(go.Scatter(
                x=[rec["e_imp"]], y=[rec["e_risk_pct"]],
                mode="markers+text", name="Recommended",
                text=["Recommended"], textposition="top center",
                textfont=dict(size=MARKER_REC, color=C_RED),
                marker=dict(size=MARKER_REC, symbol="star", color=C_RED),
                hovertemplate="Recommended<br>Expected: %{x:.0f}<br>Breakdown risk: %{y:.1f}%<extra></extra>"))

            fig_frontier.update_layout(
                title=dict(text="Performance vs Risk", xanchor="left", x=0.0, font=dict(size=32)),
                xaxis_title="Expected improvement",
                yaxis_title="Chance of breaking down (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=CHART_HEIGHT, margin=dict(l=25, r=20, t=70, b=55),
                legend=dict(font=dict(size=22), orientation="h",
                            yanchor="bottom", y=1.02, xanchor="right", x=1.0))
            fig_frontier.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
            fig_frontier.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
            style_plot_axes(fig_frontier)

        st.plotly_chart(fig_frontier, use_container_width=True,
                        config={"displayModeBar": False, "displaylogo": False})

        if rec["fallback_used"]:
            st.info(
                f"Requested posture implied {rec['risk_limit_requested_pct']:.1f}% risk limit, "
                f"but no plan reached that at {total_min} min. Showing lowest-risk plan instead.")

        # ── Recommendation summary — compact for conference ──
        st.markdown(f"""
        <div style="
            background:#EFF6FF; padding:22px 26px; border-radius:14px;
            border-left:6px solid {C_BLUE}; margin-top:12px;
        ">
          <div style="font-size:26px; font-weight:800; color:#1A1A1A; margin-bottom:12px;">
            Recommended Plan
          </div>
          <div style="font-size:24px; line-height:1.8; color:#333;">
            <strong>Easy:</strong> {rec['easy']} min &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Interval:</strong> {rec['tempo']} min &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Strength:</strong> {rec['strength']} min
          </div>
          <div style="font-size:20px; color:#4B5563; margin-top:12px; line-height:1.5;">
            {"Maximizes improvement while staying inside the risk guardrail."
             if rec["constraint_met"]
             else "No plan met the risk budget — this is the lowest-risk option available."}
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Guardrail detail"):
            st.markdown(f"""
            <div style="font-size:20px; line-height:1.7;">
                Reduce <strong>Interval by 20%</strong> if soreness stays above
                <strong>{baseline_soreness_14d:.1f}</strong> for <strong>2 consecutive days</strong>.
            </div>
            """, unsafe_allow_html=True)

# Debug
with st.expander("Debug: weekly proxy"):
    st.dataframe(df_proxy, use_container_width=True)

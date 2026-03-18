# pages/1_Scenario_Lab.py
# Option A: fixed strength minutes + risk posture defines a risk limit line
# Goal: recommended plan NEVER appears above the risk-limit line, because the optimizer enforces it.

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
    recommend_plan,   # <-- must be the updated recommend_plan(total_min,...,risk_limit_pct,fixed_strength_min,...)
)

# ---------------------------
# Global UI constants
# ---------------------------
AXIS_TITLE_FONT_SIZE = 22
AXIS_TICK_FONT_SIZE = 16
PERCENTILE_ANNOTATION_FONT_SIZE = 18
PERCENTILE_LINE_WIDTH = 3

FIXED_STRENGTH_MIN = 40         # Option A: keep constant (30–40 recommended)
FRONTIER_SAMPLES = 40
SIM_N_MAIN = 2500
SIM_N_FRONTIER = 700
SIM_N_REC = 900

# ---------------------------
# Helpers
# ---------------------------
def style_plot_axes(fig: go.Figure) -> None:
    fig.update_xaxes(
        showgrid=False,
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
    )
    fig.update_yaxes(
        showgrid=False,
        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
    )

def _min_to_hm(m: int) -> str:
    h, mm = divmod(int(m), 60)
    return f"{h}h {mm:02d}m" if h else f"{mm}m"

def belief_risk_pct_for_plan(easy: int, tempo: int, strength: int, beliefs):
    """Belief-based (load-threshold) risk; useful for hover/debug, NOT for the risk-limit line."""
    plan_load = weighted_load(easy, tempo, strength)
    risk_pct = float(sigmoid(beliefs.risk_slope * (plan_load - beliefs.risk_threshold)) * 100)
    return plan_load, risk_pct

def risk_limit_from_posture(risk_posture: float) -> float:
    """
    Audience-friendly: 'risk posture' maps to a risk budget line.
    Keep the band narrow so the chart doesn't look arbitrary.
    """
    # raw = risk_posture * 100.0
    # return float(np.clip(raw, 50.0, 70.0))
    return float(np.interp(risk_posture, [0.0, 1.0], [52.0, 62.0]))  # conservative→aggressive

# ---------------------------
# Page header
# ---------------------------
render_top_nav(active=SCENARIO_PAGE)
st.title("Scenario Lab — 10K Training Allocation Under Uncertainty")

st.markdown(
    """
    <style>
    div[data-testid="stMetricLabel"] p { font-size: 1.3rem !important; font-weight: 600 !important; line-height: 1.5 !important; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 600 !important; line-height: 1.1 !important; }
    div[data-testid="stMetricDelta"] { font-size: 1.0rem !important; font-weight: 300 !important; }

    /* Slider label */
    div[data-testid="stSlider"] > label p {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"] { transform: scaleY(1.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

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
st.subheader("What we know so far")
k1, k2, k3, k4 = st.columns(4, gap="large")

top_kpi_cards = [
    ("Consistency (last 6 weeks)", consistency_state, f"{missed_total} missed days", "#c44747", "#f3dede"),
    ("Recovery trend (last week)", recovery_state, f"Soreness {avg_soreness_last:.1f} | Sleep {avg_sleep_last:.1f}h", "#c44747", "#f3dede"),
    ("Last week disruptions", f"{missed_last} missed", "This affects uncertainty", "#c44747", "#f3dede"),
    ("Baseline soreness (14d)", f"{baseline_soreness_14d:.1f}", "Used for guardrails", "#c44747", "#f3dede"),
]


for col, (label, value, detail, detail_color, detail_bg) in zip([k1, k2, k3, k4], top_kpi_cards):
    col.markdown(
        f"""
        <div style="border:2px solid #1622a3; border-radius:0.55rem; padding:0.9rem 0.9rem 0.8rem; min-height:145px;">
            <div style="font-size:1.15rem; font-weight:600; line-height:1.3; margin-bottom:0.35rem;">{label}</div>
            <div style="font-size:2.1rem; font-weight:800; line-height:1.0; margin-bottom:0.65rem;">{value}</div>
            <div style="
                display:inline-block;
                font-size:1.25rem;
                font-weight:600;
                color:{detail_color};
                background:{detail_bg};
                border-radius:999px;
                padding:0.28rem 0.75rem;
                line-height:1.2;
            ">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
# --- constants ---
FIXED_STRENGTH_MIN = 60
MIN_EASY_MIN = 30          # prevents easy=0 (tune: 60–100)
STEP = 10

# --- init session state once ---
if "total_min" not in st.session_state:
    st.session_state.total_min = 400
if "interval_min" not in st.session_state:
    st.session_state.interval_min = 140
if "easy_min" not in st.session_state:
    st.session_state.easy_min = 220
if "last_changed" not in st.session_state:
    st.session_state.last_changed = "interval"  # or "easy"


def _reconcile_allocation():
    """
    Enforce:
      easy + interval + strength = total
      easy >= MIN_EASY_MIN
      interval >= 0
    using last_changed to decide which one stays fixed.
    """
    total = int(st.session_state.total_min)
    remaining = max(0, total - FIXED_STRENGTH_MIN)

    # If remaining is too small, clamp easy to remaining and interval to 0.
    min_easy = min(MIN_EASY_MIN, remaining)

    # If user chose a total that can't support min_easy, force total up (optional)
    # If you prefer to clamp instead of forcing total, comment these 3 lines.
    if remaining < MIN_EASY_MIN:
        total = MIN_EASY_MIN + FIXED_STRENGTH_MIN
        st.session_state.total_min = total
        remaining = total - FIXED_STRENGTH_MIN
        min_easy = MIN_EASY_MIN

    if st.session_state.last_changed == "interval":
        # interval stays, easy is derived
        interval = int(st.session_state.interval_min)
        interval = max(0, min(interval, remaining - min_easy))
        easy = remaining - interval
        easy = max(min_easy, min(easy, remaining))
        # re-derive interval in case easy was clamped
        interval = remaining - easy

    else:
        # easy stays, interval is derived
        easy = int(st.session_state.easy_min)
        easy = max(min_easy, min(easy, remaining))
        interval = remaining - easy
        interval = max(0, interval)

    st.session_state.easy_min = int(easy)
    st.session_state.interval_min = int(interval)


def _on_total_change():
    # Keep last_changed as-is; just re-balance under the new total
    _reconcile_allocation()


def _on_interval_change():
    st.session_state.last_changed = "interval"
    _reconcile_allocation()


def _on_easy_change():
    st.session_state.last_changed = "easy"
    _reconcile_allocation()


# ---- UI ----
s1, s2, s3, s4 = st.columns([1.2, 1.3, 1.3, 1.2], gap="large")

with s1:
    if "total_min" not in st.session_state:
        st.session_state.total_min = 400

    st.slider(
        "Total time (minutes)",
        min_value=120,
        max_value=480,
        value=int(st.session_state.total_min),   # <-- default value
        step=STEP,
        key="total_min",                         # <-- same widget tied to session state
        on_change=_on_total_change
    )

    total_min = int(st.session_state.total_min)
    remaining = max(0, total_min - FIXED_STRENGTH_MIN)

    st.markdown(
        f"""
        <div style="font-size:1.7rem; font-weight:400; margin-top:-0.2rem; margin-bottom:0.6rem;">
            {total_min} min <span style="font-size:1.4rem; font-weight:700; opacity:0.75;">({_min_to_hm(total_min)})</span>
        </div>
        <div style="font-size:1.25rem; font-weight:600; opacity:0.75; margin-top:-0.1rem;">
            Strength Training (minutes): {FIXED_STRENGTH_MIN} min
        </div>
        """,
        unsafe_allow_html=True,
    )

with s2:
    # dynamic bounds for interval (can't steal from min easy)
    min_easy = min(MIN_EASY_MIN, remaining)
    interval_max = max(0, remaining - min_easy)

    st.slider(
        "Interval Training (minutes)",
        min_value=0,
        max_value=int(interval_max),
        value=int(min(st.session_state.interval_min, interval_max)),
        step=STEP,
        key="interval_min",
        on_change=_on_interval_change
    )

    interval_min = int(st.session_state.interval_min)
    st.markdown(
        f"""
        <div style="font-size:1.7rem; font-weight:400; margin-top:-0.2rem; margin-bottom:0.6rem;">
            {interval_min} min <span style="font-size:1.4rem; font-weight:700; opacity:0.75;">({_min_to_hm(interval_min)})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s3:
    # dynamic bounds for easy
    min_easy = min(MIN_EASY_MIN, remaining)
    easy_max = remaining  # can be all remaining if interval goes to 0

    st.slider(
        "Easy Run (minutes)",
        min_value=int(min_easy),
        max_value=int(easy_max),
        value=int(min(max(st.session_state.easy_min, min_easy), easy_max)),
        step=STEP,
        key="easy_min",
        on_change=_on_easy_change
    )

    easy_min = int(st.session_state.easy_min)
    st.markdown(
        f"""
        <div style="font-size:1.7rem; font-weight:400; margin-top:-0.2rem; margin-bottom:0.6rem;">
            {easy_min} min <span style="font-size:1.4rem; font-weight:700; opacity:0.75;">({_min_to_hm(easy_min)})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with s4:
    risk_posture = st.slider("Risk Appetite", 0.0, 1.0, 0.35, 0.05, help="0 = conservative, 1 = aggressive")
    #risk_limit_pct = risk_limit_from_posture(risk_posture)

    st.markdown(
        f"""
        <div style="font-size:1.7rem; font-weight:400; margin-top:-0.2rem; margin-bottom:0.25rem;">
            {risk_posture:.2f} <span style="font-size:1.4rem; font-weight:700; opacity:0.75;">
            ({'Aggressive' if risk_posture >= 0.5 else 'Conservative'})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# After reconciliation, allocation is always exact
strength_min = FIXED_STRENGTH_MIN
allocated_min = int(st.session_state.easy_min) + int(st.session_state.interval_min) + strength_min

st.markdown(
    f"""
    <div style="font-size:1.6rem; font-weight:800; margin-top:-0.3rem; margin-bottom:0.4rem;">
        Total allocated: {allocated_min} / {st.session_state.total_min} min
        <span style="font-size:1.3rem; opacity:0.75;">
            ({_min_to_hm(allocated_min)} / {_min_to_hm(st.session_state.total_min)})
        </span>
        <span style='color:#0a7a2f; margin-left:0.6rem;'>✅</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# Simulation for CURRENT plan
# =========================
imps, risks = simulate_uncertainty(easy_min, interval_min, strength_min, wk_context, beliefs, n=SIM_N_MAIN, seed=999)
p10, p50, p90 = np.percentile(imps, [10, 50, 90])
e_imp = float(np.mean(imps))
e_risk_pct = float(np.mean(risks)) * 100  # 0..100

# =========================
# Main layout (charts)
# =========================
c1, c2 = st.columns([2, 2], gap="large")

with c1:
    st.subheader("Expected Impact (with Uncertainty)")

    kpi1, kpi2, kpi3 = st.columns(3)
    cards = [
        ("Typical Outcome (P50)", f"{p50:,.0f}"),
     #   ("Bad-week Outcome (P10)", f"{p10:,.0f}"),
        ("Chance of breaking down", f"{e_risk_pct:,.1f}%"),
    ]
    for col, (label, value) in zip([kpi1, kpi2, kpi3], cards):
        col.markdown(
            f"""
            <div style="font-size:1.2rem; font-weight:600; line-height:1.25; margin-bottom:0.35rem;">{label}</div>
            <div style="font-size:2.0rem; font-weight:800; line-height:1.0;">{value}</div>
            """,
            unsafe_allow_html=True,
        )

    hist_y, hist_x = np.histogram(imps, bins=35)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=hist_x[:-1], y=hist_y, name="Simulated outcomes"))

    for val, name in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        fig_dist.add_vline(
            x=float(val),
            line_width=PERCENTILE_LINE_WIDTH,
            line_dash="dash",
            annotation_text=name,
            annotation_font_size=PERCENTILE_ANNOTATION_FONT_SIZE,
        )

    fig_dist.update_layout(
        title=dict(text="Expected Performance Impact (Next 2–3 Weeks)", xanchor="left", x=0.0, font=dict(size=24)),
        xaxis_title="Improvement score (unitless)",
        yaxis_title="Simulation count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    style_plot_axes(fig_dist)
    st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

with c2:
    show_tradeoff_chart = st.toggle("Show tradeoff chart", value=False)

    if show_tradeoff_chart:
        st.subheader("Trade-offs & Recommendation")

        with st.spinner("Building trade-off frontier and recommended plan..."):

            total_min = int(st.session_state.total_min)
            strength_min = FIXED_STRENGTH_MIN
            risk_limit_pct = risk_limit_from_posture(risk_posture)

            # ---------------------------------
            # Build deterministic frontier
            # ---------------------------------
            remaining = total_min - strength_min
            min_easy = min(MIN_EASY_MIN, remaining)

            frontier_points = []
            for tempo in range(0, remaining - min_easy + 1, STEP):
                easy = remaining - tempo

                imps_s, risks_s = simulate_uncertainty(
                    easy=easy,
                    tempo=tempo,
                    strength=strength_min,
                    wk_context=wk_context,
                    beliefs=beliefs,
                    n=SIM_N_FRONTIER,
                    seed=202 + tempo,
                )

                load = weighted_load(easy, tempo, strength_min)
                belief_risk_pct = float(
                    sigmoid(beliefs.risk_slope * (load - beliefs.risk_threshold)) * 100
                )

                frontier_points.append(
                    {
                        "easy": int(easy),
                        "tempo": int(tempo),
                        "strength": int(strength_min),
                        "exp_imp": float(np.mean(imps_s)),
                        "exp_risk": float(np.mean(risks_s)) * 100.0,
                        "load": float(load),
                        "belief_risk_pct": belief_risk_pct,
                    }
                )

            frontier = pd.DataFrame(frontier_points)

            if frontier.empty:
                st.error("No candidate plans could be generated.")
                st.stop()

            # ---------------------------------
            # Recommendation
            # ---------------------------------
            rec = recommend_plan(
                total_min=total_min,
                wk_context=wk_context,
                beliefs=beliefs,
                risk_posture=risk_posture,
                risk_limit_pct=risk_limit_pct,
                fixed_strength=FIXED_STRENGTH_MIN,
                step=STEP,
                n_sims=SIM_N_REC,
                seed=123,
                min_easy_min=MIN_EASY_MIN,
            )

            if rec is None:
                st.error("Unable to generate a recommendation.")
                st.stop()

            # use the actual line associated with the recommendation
            risk_limit_used_pct = float(rec["risk_limit_used_pct"])

            # ---------------------------------
            # Plot
            # ---------------------------------
            fig_frontier = go.Figure()
            fig_frontier.update_layout(
                title="Performance vs Risk Trade-off",
                xaxis_title="Expected improvement (mean)",
                yaxis_title="Chance of breaking down (%)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(l=20, r=20, t=60, b=20),
            )

            line_label = (
                f"Risk limit = {risk_limit_used_pct:.1f}%"
                if rec["constraint_met"]
                else f"Minimum achievable risk = {risk_limit_used_pct:.1f}%"
            )

            fig_frontier.add_hline(
                y=risk_limit_used_pct,
                line_dash="dash",
                line_width=3,
                line_color="rgba(180, 0, 0, 0.7)",
                annotation_text=line_label,
                annotation_position="top left",
                annotation_font_size=20,
                annotation_font_color="rgba(180, 0, 0, 0.9)",
            )

            fig_frontier.add_trace(
                go.Scatter(
                    x=frontier["exp_imp"],
                    y=frontier["exp_risk"],
                    mode="markers",
                    name="Candidate plans",
                    customdata=np.stack(
                        [frontier["load"], frontier["belief_risk_pct"]], axis=1
                    ),
                    text=[
                        f"Easy {r.easy} | Interval {r.tempo} | Strength {r.strength}"
                        for r in frontier.itertuples(index=False)
                    ],
                    hovertemplate=(
                        "%{text}"
                        "<br>Expected: %{x:.0f}"
                        "<br>Chance breakdown: %{y:.1f}%"
                        "<br>Load: %{customdata[0]:.0f}"
                        "<br>Belief risk@load: %{customdata[1]:.1f}%"
                        "<extra></extra>"
                    ),
                    marker=dict(
                        size=9,
                        color="#7DD3FC",
                        opacity=0.45,
                        line=dict(width=1.0, color="rgba(0,0,0,0.8)"),
                    ),
                )
            )

            fig_frontier.add_trace(
                go.Scatter(
                    x=[e_imp],
                    y=[e_risk_pct],
                    mode="markers+text",
                    name="Current",
                    text=["Current"],
                    textposition="top center",
                    textfont=dict(size=14, color="#06613e"),
                    marker=dict(size=16, symbol="diamond", color="#06613e"),
                    hovertemplate=(
                        "Current"
                        "<br>Expected: %{x:.0f}"
                        "<br>Chance breakdown: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                )
            )

            fig_frontier.add_trace(
                go.Scatter(
                    x=[rec["e_imp"]],
                    y=[rec["e_risk_pct"]],
                    mode="markers+text",
                    name="Recommended",
                    text=["Recommended"],
                    textposition="top center",
                    textfont=dict(size=14, color="#EB3604"),
                    marker=dict(size=18, symbol="star", color="#EB3604"),
                    hovertemplate=(
                        "Recommended"
                        "<br>Expected: %{x:.0f}"
                        "<br>Chance breakdown: %{y:.1f}%"
                        "<extra></extra>"
                    ),
                )
            )

            style_plot_axes(fig_frontier)

        st.plotly_chart(
            fig_frontier,
            use_container_width=True,
            config={"displayModeBar": False, "displaylogo": False},
        )

        if rec["fallback_used"]:
            st.info(
                f"The requested posture implied a risk limit of {rec['risk_limit_requested_pct']:.1f}%, "
                f"but no plan could reach that at {total_min} total minutes. "
                f"Showing the lowest-risk available plan instead."
            )

        st.markdown("### Recommended Training Plan")
        st.markdown(
            f"""
            <div style="font-size:1.2rem; line-height:1.7;">
                <strong>Easy:</strong> {rec['easy']} min<br>
                <strong>Interval:</strong> {rec['tempo']} min<br>
                <strong>Strength:</strong> {rec['strength']} min
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:1.2rem; font-weight:700; margin-top:0.5rem;">Why this plan</div>',
            unsafe_allow_html=True,
        )

        why_text = (
            """
            <div style="font-size:1.1rem; line-height:1.7;">
                <ul style="margin-top:0.4rem;">
                    <li>Maximizes expected improvement while staying inside the requested risk budget</li>
                    <li>Balances performance gain against breakdown risk</li>
                </ul>
            </div>
            """
            if rec["constraint_met"]
            else
            """
            <div style="font-size:1.1rem; line-height:1.7;">
                <ul style="margin-top:0.4rem;">
                    <li>No plan met the requested risk budget at this total time</li>
                    <li>This is the lowest-risk plan available while preserving the fixed 40-minute strength block</li>
                </ul>
            </div>
            """
        )
        st.markdown(why_text, unsafe_allow_html=True)

        st.markdown(
            '<div style="font-size:1.2rem; font-weight:700; margin-top:0.5rem;">Guardrail</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:1.1rem; line-height:1.7;">
                Reduce <strong>Interval by 20%</strong> if soreness stays above <strong>{baseline_soreness_14d:.1f}</strong>
                for <strong>2 consecutive days</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
# Debug only (keep collapsed for conference)
with st.expander("Debug: weekly proxy"):
    st.dataframe(df_proxy, use_container_width=True)
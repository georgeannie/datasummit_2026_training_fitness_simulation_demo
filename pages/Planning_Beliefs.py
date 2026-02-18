# planning_beliefs.py
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import date, timedelta
from utils.nav import SCENARIO_PAGE, BELIEFS_PAGE, render_top_nav
from core import (
    get_history_and_beliefs,
    sat_gain,
    sigmoid,
    weighted_load,
    DEFAULT_H,
    CARRYOVER_MULTIPLIER,
    BASE_PRIOR

)
def belief_metric(col, title, value, delta=None, delta_color=None,
                   info_md=None):
    # st.markdown("""
    # <style>
    # /* Streamlit bordered container wrapper */
    # div[data-testid="stVerticalBlockBorderWrapper"]{
    # border: 2px solid #2E3B8F !important;
    # border-radius: 14px !important;
    # background: #f8fafc !important;
    # box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    # }
    # </style>
    # """, unsafe_allow_html=True)

    with col:
        with st.container(border=True):
            st.markdown("""
            <style>
            /* Streamlit bordered container wrapper */
            border: 2px solid #2E3B8F !important;
            border-radius: 14px !important;
            background: #f8fafc !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
            </style>
            """, unsafe_allow_html=True)
            # Header row
            header = st.columns([0.85, 0.15])
            with header[0]:
                st.markdown(f"**{title}**")
            with header[1]:
                if info_md:
                    with st.popover("ⓘ"):
                        st.markdown(info_md)

            # Metric
            st.metric(
                label="Consistency",
                value=value,
                delta=delta,
                delta_color=delta_color,
                label_visibility="hidden"
            )

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Planning Beliefs — Learned from 6 Weeks", layout="wide",initial_sidebar_state="collapsed")
PRESENTATION_CSS = """
<style>

/* Cards (metrics, info boxes) */
div[data-testid="stMetric"], 
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"] {
}

/* Markdown blockquote "belief cards" */
blockquote {
  background: #0B1220;              /* deep navy */
  border-left: 10px solid #60A5FA;  /* bright blue */
  color: #F9FAFB;                   /* near-white */
  padding: 1.2rem 1.4rem;
  border-radius: 14px;
  font-size: 16px;
  font-weight: 520;
  font-family: "Source Sans Pro", sans-serif;
  font-style: normal;
  line-height: 1.6;
  box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

blockquote strong {
  display: block;
  font-size: 17px;
  font-weight: 750;
  margin-bottom: 0.35rem;
  font-color: rgba(0, 0, 0, 0.1);
  color: #FFFFFF; /* pure black for emphasis */
}


/* Expanders */
div[data-testid="stExpander"] {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.10);
}


/* Make captions readable */
large, .stCaption {
  color: rgba(248, 250, 252, 0.78) !important;
  font-size: 20px;
  font-weight: 1000;
}

</style>
"""
header_style = """
<style>
    /* Specific selector for the st.dataframe header cells */
    .stDataFrame thead th {
        background-color: black !important;
        color: white !important; /* Make text white for visibility */
    }
</style>
"""
st.markdown(PRESENTATION_CSS, unsafe_allow_html=True)
render_top_nav(active=BELIEFS_PAGE)
st.title("Planning Beliefs — Learned from 6 Weeks of Training")

st.markdown(
    "**Purpose:** Make assumptions explicit **and learned from history** — so we can challenge them, test them, and update them."
)
df_daily, df_proxy, beliefs = get_history_and_beliefs(seed=11, weeks=6)
params = beliefs.params

# =============================
# Belief 1 — Diminishing returns (learned response curves)
# =============================
st.subheader("Belief 1 — More Training Helps, Until It Doesn’t (Diminishing Returns)")

left1, right1 = st.columns([1.6, 1.0], gap="large")
with left1:
    x = np.arange(0, 300, 10)
    # fig1 = go.Figure()
    # fig1.add_trace(go.Scatter(x=x, y=[sat_gain(m, **params["easy"]) for m in x], mode="lines", name="Easy"))
    # fig1.add_trace(go.Scatter(x=x, y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x], mode="lines", name="Tempo"))
    # fig1.add_trace(go.Scatter(x=x, y=[sat_gain(m, **params["strength"]) for m in x], mode="lines", name="Strength"))

    # fig1.update_layout(
    #     title="Response Curves Learned from History (Training Gain vs Weekly Minutes)",
    #     xaxis_title="Weekly minutes (per lever)",
    #     yaxis_title="Training gain (unitless)",
    #     height=550,
    #     width=350,
    #     margin=dict(l=20, r=20, t=60, b=20),
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     template="plotly_dark",
    #     paper_bgcolor="rgba(0,0,0,0)",
    #     plot_bgcolor="rgba(0,0,0,0)",
    #     font=dict(size=18),
    # )
    wk_easy = df_proxy["easy_min"].values
    wk_tempo = df_proxy["moderate_run_comfort_pace_min"].values
    wk_strength = df_proxy["strength_min"].values

    fig1 = go.Figure()

    # Strong color palette for light background
    colors = {
        "easy": "#1f77b4",      # blue
        "tempo": "#d62728",     # red
        "strength": "#2ca02c"   # green
    }

    # ---- Curves (thicker & darker) ----
    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["easy"]) for m in x],
        mode="lines",
        name="Easy",
        line=dict(color=colors["easy"], width=4)
    ))

    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x],
        mode="lines",
        name="Tempo",
        line=dict(color=colors["tempo"], width=4)
    ))

    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["strength"]) for m in x],
        mode="lines",
        name="Strength",
        line=dict(color=colors["strength"], width=4)
    ))

    # ---- Observed Points (filled + bold) ----
    fig1.add_trace(go.Scatter(
        x=wk_easy,
        y=[sat_gain(m, **params["easy"]) for m in wk_easy],
        mode="markers",
        name="Observed (Easy)",
        marker=dict(size=12, color=colors["easy"], line=dict(width=1.5, color="black"))
    ))

    fig1.add_trace(go.Scatter(
        x=wk_tempo,
        y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in wk_tempo],
        mode="markers",
        name="Observed (Tempo)",
        marker=dict(size=12, color=colors["tempo"], line=dict(width=1.5, color="black"))
    ))

    fig1.add_trace(go.Scatter(
        x=wk_strength,
        y=[sat_gain(m, **params["strength"]) for m in wk_strength],
        mode="markers",
        name="Observed (Strength)",
        marker=dict(size=12, color=colors["strength"], line=dict(width=1.5, color="black"))
    ))

    # ---- Layout (light theme + clean legend) ----
    fig1.update_layout(
        #title="Response Curves Learned from History (Training Gain vs Weekly Minutes)",
        xaxis_title="Weekly minutes (per lever)",
        yaxis_title="Training gain (unitless)",
        height=550,
        margin=dict(l=10, r=20, t=80, b=40),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",   # transparent outer background
        plot_bgcolor="rgba(0,0,0,0)",    # transparent plot area
        font=dict(size=20),
        title=dict(
            text="Response Curves Learned from History <br> (Training Gain vs Weekly Minutes)",
            x=0.5,
            xanchor="right",
            font=dict(size=20)
        ),

        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=1.15,          # Push higher so it’s visible
            xanchor="center",
            x=8.5
        )
    )
    fig1.update_layout(
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)")
)


    st.plotly_chart(fig1, width='stretch')

 
with right1:
    params_table = pd.DataFrame.from_dict(
    {
        "easy": {
            "Peak Benefit (α)": params["easy"]["alpha"],
            "Diminishing Point (k)": params["easy"]["k"],
            "Shape (h)": params["easy"]["h"],
        },
        "Tempo (moderate)": {
            "Peak Benefit (α)": params["moderate_run_comfort_pace"]["alpha"],
            "Diminishing Point (k)": params["moderate_run_comfort_pace"]["k"],
            "Shape (h)": params["moderate_run_comfort_pace"]["h"],
        },
        "strength": {
            "Peak Benefit (α)": params["strength"]["alpha"],
            "Diminishing Point (k)": params["strength"]["k"],
            "Shape (h)": params["strength"]["h"],
        },
    },
        orient="index",
    ).reset_index().rename(columns={"index": "Training Lever"})

    # round for presentation
    params_table["Peak Benefit (α)"] = params_table["Peak Benefit (α)"].round(1)
    params_table["Diminishing Point (k)"] = params_table["Diminishing Point (k)"].round(1)
    #params_table["Shape (h)"] = params_table["Shape (h)"].round(2)

    st.markdown("""
        <div style="
            background-color: #dae7f5;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2E3B8F;
            color: #1f2937;
        ">
        <h4 style="margin-bottom:10px;">Belief Card</h4>
        <p style="font-size: 18px"><strong>Not all training minutes are equal.</strong></p>
        <p style="font-size: 18px">The first minutes of effort create most of the benefit. Beyond a point, additional time adds fatigue faster than performance.</p>
        
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 18px; opacity: 0.7;">What the data reinforces</p>
    """, unsafe_allow_html=True)
    st.dataframe(
        params_table,
        width='stretch',
        hide_index=True,
    )
    st.markdown("Implication: The best plan is usually a reallocation. Chase marginal gains, not total effort.")

st.markdown("---")

# =============================
# Belief 2 — Carryover (learned half-life)
# =============================
st.subheader("Belief 2 — Training Effects Persist, Then Fade (Carryover & Decay)")
# st.caption(
#     "Estimating how much last week’s training still helps this week. In other words, how long fitness carries over.")
# st.caption("Step 1: Fit a model to weekly training data to estimate carryover coefficient (c).")
# st.caption(" Carryover Coefficient (c) = This week’s improvement / Last week’s training improvement.") 
# st.caption("Higher c means last week’s training matters more for this week’s improvement.")
# st.caption("Step 2: Find the decay of training benefits over time.")
# st.caption("This tells how long does it take for something that shrinks by a factor of c each week to be cut in half?. For example, a half-life of 1 week means that after 1 week, the training benefit is reduced by half.")

st.caption("We estimate how much last week’s training still helps this week — i.e., how long fitness carries over.")

left2, right2 = st.columns([1.6, 1.0], gap="large")
with left2:
    # show decay curve in days using learned half-life in weeks (approx)
    half_life_days = 7.0 * beliefs.half_life_wks if beliefs.half_life_wks > 0 else 4.0
    days = np.arange(0, 15, 1)
    decay = np.exp(-np.log(2) * (days / max(1e-6, half_life_days)))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=days, y=decay, mode="lines", name="Remaining benefit",
                              line=dict(color="#2E3B8F", width=4))
    )
#    fig2.add_vline(x=2, line_dash="dash", annotation_text="~Tue if trained Sun")
    fig2.add_vline(x=float(half_life_days), line_dash="dash", annotation_text="Half-life (learned)")
    fig2.update_layout(
         title=dict(
            text="Carryover Curve (Benefit Remaining vs Days Since Workout)",
            x=0.5,
            xanchor="right",
            font=dict(size=20)
        ),
        xaxis_title="Days since workout",
        yaxis_title="Remaining benefit (0–1)",
        height=550,
        width=350,
        paper_bgcolor="rgba(0,0,0,0)",   # transparent outer background
        plot_bgcolor="rgba(0,0,0,0)",    # transparent plot area
        margin=dict(l=20, r=20, t=60, b=20),
        showlegend=False,
    )
    fig2.add_vline(
        x=float(half_life_days),
        line_width=5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.7)",
        annotation_text="Half-life"
    )
    st.plotly_chart(fig2, width='stretch')
  
with right2:
    st.markdown("""
    <div style="
        background-color: #dae7f5;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E3B8F;
        color: #1f2937;
    ">
    <h4 style="margin-bottom:14px;">Belief Card</h4>

    <p style="font-size: 18px"><strong>Training works over time, not instantly.</strong></p>

    <p style="font-size: 18px">Weekly decisions influence future weeks. Timing matters, not just totals.</p>

    <p style="font-size: 18px; opacity: 0.8; margin-top: 12px;">
    <b>Model Parameter: </b
    <strong><b>Carryover (c)</b></strong> — controls how long effects persist.
    </p>

    </div>
    """, unsafe_allow_html=True)

   # st.subheader("Carryover — how long benefits last")

    c1, c2 = st.columns(2, gap="large")

    belief_metric(
        c1,
        title="Weekly benefit retained",
        value="4%",
        delta="Fades quickly",
        delta_color="inverse",
#        delta_arrow="off",
        info_md="""
    Fraction of last week's training benefit that persists into the next week.
    Low values mean benefits must be reinforced consistently.
   
**How retention is measured**  
Observe how performance-related signals change after a training week, then measure how much of that improvement remains when training load drops.

**Decision implication**  
Missing a week costs more than slightly under-training in a week.
    """
    )

    belief_metric(
        c2,
        title="Half-life of benefit",
        value="0.21 weeks",
        delta="~1–2 days",
        delta_color="normal",
   #     delta_arrow="auto",
        info_md="""
    Time until 50% of remaining benefit decays.
    Short half-life implies timing matters more than volume.
    """
    )
    st.markdown(
    "Implication: Missing a week costs more than missing a single workout."
)

st.markdown("---")

# =============================
# Belief 3 — Risk threshold (learned)
# =============================
st.subheader("Belief 3 — Risk Accelerates After a Threshold")

left3, right3 = st.columns([1.6, 1.0], gap="large")
with left3:
    loads = np.arange(0, 801, 10)
    risk_curve = sigmoid(beliefs.risk_slope * (loads - beliefs.risk_threshold)) * 100

    fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(x=loads, y=risk_curve, mode="lines", name="Risk probability",
    #                           #line=dict(color="#2E3B8F", width=4)
    #                           )
    # )

    # plot observed weekly loads as markers
    wk = df_proxy.copy()
    wk["load"] = weighted_load(wk["easy_min"], wk["moderate_run_comfort_pace_min"], wk["strength_min"])
    wk["risky_week_proxy"] = ((wk["avg_soreness"] >= 5.2) | (wk["missed_days"] >= 1)).astype(int)

   # wk_bad = wk[wk["risky_week_proxy"] == 1]
    fig3.add_trace(go.Scatter(
        x=wk["load"],
        y=wk["risky_week_proxy"] * 100,
        # x=wk_bad["load"],
        # y=[75]*len(wk_bad), 
        mode="markers", 
        marker=dict(
            size=16,
#            color="#ff4d4d",
            opacity=0.9,
            line=dict(width=2.5, color="#7f0000")
        ),
        name="Observed risky weeks (proxy)",
        hovertemplate="Load: %{x:.0f}<br>Risky proxy: %{y:.0f}%<extra></extra>"
    ))

    fig3.update_layout(
        title=dict(
            text="Where Training Plans Break Down (Risk vs Training Load)",
            x=0.5,
            xanchor="right",
            font=dict(size=20)
        ),
        xaxis_title="Training Load This Week (weighted minutes)",
        yaxis_title="Risk (%) - Chance the Week Falls Apart",
        height=550,
        width=350,
         paper_bgcolor="rgba(0,0,0,0)",   # transparent outer background
        plot_bgcolor="rgba(0,0,0,0)",    # transparent plot area
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="v", yanchor="bottom", y=0.5, xanchor="right", x=1),
    )
    fig3.add_vline(
    x=beliefs.risk_threshold,
    line_dash="dash",
    line_color="rgba(255,255,255,0.7)",
    annotation_text="Danger Zone",
    annotation_position="top"
    )
    fig3.add_vrect(
    x0=beliefs.risk_threshold,
    x1=max(loads),
    fillcolor="rgba(255, 0, 0, 0.08)",
    line_width=0
)
    st.plotly_chart(fig3, width='stretch')
   
with right3:
    st.markdown("""
    <div style="
        background-color: #dae7f5;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E3B8F;
        color: #1f2937;
    ">
    <h4 style="margin-bottom:10px;">Belief Card</h4>

    <p style="font-size: 18px"><strong>Risk rises faster than performance.</strong></p>

    <p style="font-size: 18px">Aggressive plans look strong on average, but fail because of tail risk. Guardrails matter.</p>

    <p style="font-size: 18px; margin-top: 10px;">
    <strong>Risk Threshold</strong> — defines the safe operating zone
    </p>

    </div>
    """, unsafe_allow_html=True)


    m1, m2 = st.columns(2, gap="large")

    belief_metric(
        m1,
        title="Danger zone threshold (weighted load)",
        value=f"{int(beliefs.risk_threshold)}",
        delta="Bad weeks accelerate",
        delta_color="inverse",
        info_md="""
    **What this means**  
    This is the load where the probability of a “week falling apart” starts rising rapidly  
    (missed sessions and/or high soreness).

    **How it’s used**  
    Scenario Lab avoids plans that sit near or above this threshold when you choose a conservative posture.
    """
    )

    belief_metric(
        m2,
        title="Risk acceleration",
        value=f"{beliefs.risk_slope:.3f}",
        delta="Steeper = risk climbs faster",
        delta_color="normal",
        info_md="""
    **What this means**  
    How quickly risk increases once you cross the threshold.

    - Higher value → risk spikes sharply (cliff)
    - Lower value → risk increases more gradually

    **Why it matters**  
    This is the “tail risk” driver: aggressive plans may look good on average but fail more often.
    """
    )

    st.markdown("Implication: optimize for robustness, not just the best average plan.")
st.markdown("---")

st.subheader("How This Connects Back to Scenario Lab")
st.markdown("""
<div style="
    background-color: #e8f2ff;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #2E3B8F;
    font-size: 18px;
    line-height: 1.6;
    color: #1f2937;
">
These beliefs are learned from the last 6 weeks and power the <strong>Scenario Lab.</strong><br><br>

If you disagree with a curve, you change the belief — then re-evaluate the decision.<br><br>

<strong>MMM isn’t a forecast. It’s a way to make assumptions explicit so decisions can be stress-tested under uncertainty.</strong>
</div>
""", unsafe_allow_html=True)
with st.expander("Debug: weekly proxy table"):
    st.dataframe(df_proxy, width='stretch')
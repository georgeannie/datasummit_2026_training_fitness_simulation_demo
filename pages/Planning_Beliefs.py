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
AXIS_TITLE_FONT_SIZE = 22
AXIS_TICK_FONT_SIZE = 16


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


# @st.dialog("Details")
# def show_info_dialog(title: str, info_md: str) -> None:
#     st.markdown(f"### {title}")
#     st.markdown(info_md)


def _info_key(title: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in title)


def belief_metric(col, title, value, delta=None, delta_color=None,
                   info_md=None):
    with col:
        with st.container(border=True):
            st.markdown("""
            <style>
            /* Streamlit bordered container wrapper */
            border: 2px solid #2E3B8F !important;
            border-radius: 15px !important;
            background: #f8fafc !important;
            box-shadow: 1 4px 10px rgba(0,0,0,0.06);
            div[data-testid="stMetricValue"] > div {
                font-size: 50px; /* Adjust this value */
            }
            </style>
            """, unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:22px; font-weight:600; line-height:1.2; margin:0 3rem 0.6rem 0;">{title}</div>',
                unsafe_allow_html=True,
            )
            # if info_md:
            #     if st.button("❔", key=f"info_{_info_key(title)}", help="More information"):
            #         show_info_dialog(title, info_md)

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
PRESENTATION_CSS = """
<style>

/* Cards (metrics, info boxes) */
div[data-testid="stMetric"], 
div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"] {
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  position: relative;
}

div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 5;
  width: auto;
}

div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stButton"] > button {
  width: 34px !important;
  min-width: 34px !important;
  height: 34px !important;
  min-height: 34px !important;
  border-radius: 999px !important;
  padding: 0 !important;
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

# st.markdown(
#     "**Purpose:** Make assumptions explicit **and learned from history** — so we can challenge them, test them, and update them."
# )
df_daily, df_proxy, beliefs = get_history_and_beliefs(seed=11, weeks=6)
params = beliefs.params

# =============================
# Belief 1 — Carryover (learned half-life)
# =============================
st.subheader("Belief 1 — Training Effects Persist, Then Fade (Carryover & Decay)")
# st.caption(
#     "Estimating how much last week’s training still helps this week. In other words, how long fitness carries over.")
# st.caption("Step 1: Fit a model to weekly training data to estimate carryover coefficient (c).")
# st.caption(" Carryover Coefficient (c) = This week’s improvement / Last week’s training improvement.") 
# st.caption("Higher c means last week’s training matters more for this week’s improvement.")
# st.caption("Step 2: Find the decay of training benefits over time.")
# st.caption("This tells how long does it take for something that shrinks by a factor of c each week to be cut in half?. For example, a half-life of 1 week means that after 1 week, the training benefit is reduced by half.")

#st.caption("We estimate how much last week’s training still helps this week — i.e., how long fitness carries over.")

left2, right2 = st.columns([1.6, 1.0], gap="large")
with left2:
    # Convert learned half-life from weeks to days
    half_life_days = 7.0 * beliefs.half_life_wks if beliefs.half_life_wks > 0 else 4.0

    days = np.arange(0, 15, 1)
    decay = np.exp(-np.log(2) * (days / max(1e-6, half_life_days)))

    # Reference points for readability
    day1 = 1
    day7 = 7
    y_day1 = float(np.exp(-np.log(2) * (day1 / max(1e-6, half_life_days))))
    y_day7 = float(np.exp(-np.log(2) * (day7 / max(1e-6, half_life_days))))

    fig2 = go.Figure()

    # Main curve
    fig2.add_trace(go.Scatter(
        x=days, y=decay,
        mode="lines",
        name="Benefit remaining",
        line=dict(color="#2E3B8F", width=4),
        hovertemplate="Day %{x}<br>Benefit left: %{y:.0%}<extra></extra>",
        showlegend=False
    ))

    # Half-life line + clearer annotation
    fig2.add_vline(
        x=float(half_life_days),
        line_color="rgba(17,24,39,0.75)",
        line_width=4,
        line_dash="dash",
    )
    fig2.add_annotation(
        x=float(half_life_days),
        y=0.6,
        xref="x", yref="y",
        text="Half-life: benefit drops to 50% here",
        showarrow=True,
        arrowhead=2,
        ax=140, ay=-100,
        font=dict(size=22, color="rgba(50,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.92)",
        #bordercolor="rgba(17,24,39,0.20)",
        #borderwidth=1
    )

    # Reference markers (Day 1 and Day 7)
    fig2.add_trace(go.Scatter(
        x=[day1, day7],
        y=[y_day1, y_day7],
        mode="markers+text",
        marker=dict(size=16, color="rgba(46,59,143,0.8)", line=dict(width=1.2, color="rgba(17,24,39,0.55)")),
        text=[f"Day 1: {y_day1:.0%}", f"Day 7: {y_day7:.0%}"],
        textposition=["top center", "top center"],
        textfont=dict(size=25,  color="rgba(150,24,19,0.95)"),
        hovertemplate="Day %{x}<br>Benefit left: %{y:.0%}<extra></extra>",
        showlegend=False
    ))

    fig2.update_layout(
        title=dict(
            text="Carryover Curve",
            xanchor="left",
            x=0.0,
            font=dict(size=26)
        ),
        # Plain-language subtitle (under title)
        # annotations=fig2.layout.annotations + (
        #     go.layout.Annotation(
        #         x=-0.12, y=1.1, xref="paper", yref="paper",
        #         #text="How much the last workout still helps as days pass",
        #         showarrow=False,
        #         font=dict(size=18, color="rgba(17,24,39,0.9)"),
        #         align="left"
        #     ),
        # ),
        xaxis_title="Days since workout",
        yaxis_title="Benefit left (%)",
        height=520,
        width=520,  # give the plot room; avoid cramped look at 350
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=25, r=20, t=80, b=45),
        showlegend=False,
        template="plotly_white"
    )

    # Demo-friendly ranges and light grid
    fig2.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig2.update_xaxes(range=[0, 14])
    fig2.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")

    style_plot_axes(fig2)
    # Streamlit expects use_container_width instead of a width string
    st.plotly_chart(fig2, width='stretch',
                    config={
        "displayModeBar": False,   # hides the toolbar entirely
        "scrollZoom": False,
        "doubleClick": "reset",
        "displaylogo": False
    })
  
with right2:
    st.markdown("""
    <div style="
        background-color: #dae7f5;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E3B8F;
        color: #111827;
    ">
      <h4 style="margin-bottom:14px; font-size:26px">Belief Card</h4>

      <p style="font-size: 23px; font-weight: 2; margin:0 0 8px 0;"><strong>Training works over time, not instantly.</strong></p>
      <p style="font-size: 23px; margin:0 0 12px 0;">What you do this week still helps next week—but it fades.</p>

      <p style="font-size: 24px; opacity: 0.85; margin:0;">
        <strong>Carryover (c)</strong> — controls how long effects persist.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    # KPI 1: Benefit left after 7 days (ties directly to curve)
    belief_metric(
        c1,
        title="Benefit left after 7 days",
        value=f"{y_day7:.0%}",
        delta="Fades quickly",
        delta_color="inverse",
        info_md=f"""
**What this means**  
After a week, about **{y_day7:.0%}** of the benefit remains.

**Why it matters**  
If benefits fade fast, consistency matters more than one heroic week.
"""
    )

    # KPI 2: Half-life shown in DAYS (not weeks)
    half_life_days_display = float(half_life_days)
    belief_metric(
        c2,
        title="Half-life of benefit   ",
        value=f"{half_life_days_display:.1f} days",
        delta="~1–2 days" if half_life_days_display <= 2.0 else "Fades over days",
        delta_color="normal",
        info_md=f"""
**What this means**  
Half-life is the time until benefit drops to **50%** (≈ **{half_life_days_display:.1f} days**).

**Why it matters**  
Short half-life means timing and spacing matter more than total volume.
"""
    )

 #   st.markdown("**Implication:** consistency beats single heroic days.")

st.markdown("---")

# =============================
# Belief 2 — Diminishing returns (learned response curves)
# =============================
st.subheader("Belief 2 — More Training Helps, Until It Doesn’t (Diminishing Returns)")

left1, right1 = st.columns([1.6, 1.0], gap="large")
with left1:
    x = np.arange(0, 300, 5)  # smoother curves for stage

    wk_easy = df_proxy["easy_min"].values
    wk_interval = df_proxy["moderate_run_comfort_pace_min"].values
    wk_strength = df_proxy["strength_min"].values

    fig1 = go.Figure()

    colors = {
        "easy": "#1f77b4",      # blue
        "interval": "#d62728",  # red
        "strength": "#2ca02c"   # green
    }

    # --- Lines (legend = ON) ---
    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["easy"]) for m in x],
        mode="lines",
        name="Easy",
        line=dict(color=colors["easy"], width=4),
        hovertemplate="Easy<br>Minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=True
    ))

    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x],
        mode="lines",
        name="Interval",
        line=dict(color=colors["interval"], width=4),
        hovertemplate="Interval<br>Minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=True
    ))

    fig1.add_trace(go.Scatter(
        x=x,
        y=[sat_gain(m, **params["strength"]) for m in x],
        mode="lines",
        name="Strength",
        line=dict(color=colors["strength"], width=4),
        hovertemplate="Strength<br>Minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=True
    ))

    # --- Observed points (legend = OFF, lighter styling) ---
    fig1.add_trace(go.Scatter(
        x=wk_easy,
        y=[sat_gain(m, **params["easy"]) for m in wk_easy],
        mode="markers",
        marker=dict(
            size=11,
            color="rgba(31,119,180,0.55)",  # lighter
            line=dict(width=1.2, color="rgba(17,24,39,0.65)")
        ),
        hovertemplate="Observed week<br>Easy minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=False
    ))

    fig1.add_trace(go.Scatter(
        x=wk_interval,
        y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in wk_interval],
        mode="markers",
        marker=dict(
            size=11,
            color="rgba(214,39,40,0.60)",
            line=dict(width=1.2, color="rgba(17,24,39,0.65)")
        ),
        hovertemplate="Observed week<br>Interval minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=False
    ))

    fig1.add_trace(go.Scatter(
        x=wk_strength,
        y=[sat_gain(m, **params["strength"]) for m in wk_strength],
        mode="markers",
        marker=dict(
            size=11,
            color="rgba(44,160,44,0.55)",
            line=dict(width=1.2, color="rgba(17,24,39,0.65)")
        ),
        hovertemplate="Observed week<br>Strength minutes: %{x:.0f}<br>Lift: %{y:.0f}<extra></extra>",
        showlegend=False
    ))

    # --- Interval flattening marker (k) ---
    k_interval = float(params["moderate_run_comfort_pace"]["k"])

    # Very subtle shading to the right of k (interval only)
    fig1.add_vrect(
        x0=k_interval, x1=max(x),
        fillcolor="rgba(214,39,40,0.04)",  # lighter than before
        line_width=0,
        layer="below"
    )

    fig1.add_vline(
        x=k_interval,
        line_dash="dash",
        line_width=3,
        line_color="rgba(17,24,39,0.60)"
    )

    fig1.add_annotation(
        x=k_interval,
        y=max([sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x]) * 0.96,
        text=f"Interval Starts flattening ~{int(round(k_interval))} min",
        showarrow=True,
        arrowhead=2,
        ax=35, ay=-25,
        font=dict(size=20, color="rgba(17,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(17,24,39,0.20)",
        borderwidth=1
    )

    # --- Layout (clean + demo readable) ---
    fig1.update_layout(
        title=dict(
            text="Response Curves: Minutes → Lift",
            xanchor="left",
            x=0.0,
            font=dict(size=26)
        ),
        xaxis_title="Minutes per week (by workout type)",
        yaxis_title="Improvement score",
        height=520,
        margin=dict(l=20, r=20, t=70, b=45),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=18),
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.75,
            xanchor="right",
            font=dict(size=23),
            x=0.2
        ),
    )

    # light gridlines (demo)
    fig1.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")

    style_plot_axes(fig1)

    st.plotly_chart(fig1, width='stretch',
                    config={
        "displayModeBar": False,   # hides the toolbar entirely
        "scrollZoom": False,
        "doubleClick": "reset",
        "displaylogo": False
    })
 
with right1:
    params_table = pd.DataFrame([
        {
            "Lever": "Easy",
            "Max lift": int(round(float(params["easy"]["alpha"]), 0)),
            "Starts flattening (min)": int(round(float(params["easy"]["k"]), 0)),
        },
        {
            "Lever": "Interval",
            "Max lift": int(round(float(params["moderate_run_comfort_pace"]["alpha"]), 0)),
            "Starts flattening (min)": int(round(float(params["moderate_run_comfort_pace"]["k"]), 0)),
        },
        {
            "Lever": "Strength",
            "Max lift": int(round(float(params["strength"]["alpha"]), 0)),
            "Starts flattening (min)": int(round(float(params["strength"]["k"]), 0)),
        },
    ])

    st.markdown("""
        <div style="
            background-color: #eae7f5;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #6902c4;
            color: #111827;
        ">
        <h4 style="margin-bottom:12px;font-size:26">Belief Card</h4>
        <p style="font-size: 23px; margin:0 0 8px 0;"><strong>Not all training minutes are equal.</strong></p>
        <p style="font-size: 23px; margin:0;">Early minutes buy the most lift. After a point, extra time adds less benefit.</p>
        </div>
    """, unsafe_allow_html=True)

    # st.markdown("""<p style="font-size: 18px; opacity: 0.7; margin-top:10px;">What the data reinforces</p>""",
    #             unsafe_allow_html=True)

    # Best if you need larger font styling
    st.markdown(" ")
    styled = (
    params_table.style
    .set_properties(**{
        "font-size": "21px",
        "font-weight": "600",
        "text-align": "left"
    })
    .set_table_styles([
        {
            "selector": "table",
            "props": [
                ("font-size", "21px"),
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ],
        },
        {
            "selector": "th",
            "props": [
                ("font-size", "21px"),
                ("font-weight", "1000"),
                ("background-color", "#e8e1ee"),
                ("color", "white"),
                ("padding", "10px 10px"),
                ("text-align", "right"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("font-size", "21px"),
                ("font-weight", "600"),
                ("padding", "10px 40px"),
                ("border-color", "grey"),
                  ("text-align", "right"),
            ],
        },
        {
            "selector": "tbody th",
            "props": [
                ("font-size", "21px"),
                ("font-weight", "700"),
                ("background-color", "white"),
                ("color", "white"),
                ("padding", "18px 16px"),
                  ("text-align", "right"),
            ],
        },
    ])
    .hide(axis="index")
)

    st.markdown(styled.to_html(), unsafe_allow_html=True)

   # st.markdown("**Implication:** reallocate minutes toward the highest marginal lift—don’t just add more time.")
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

    wk["is_bad_week"] = wk["risky_week_proxy"].astype(bool)

    wk_bad = wk[wk["is_bad_week"]]
    wk_ok  = wk[~wk["is_bad_week"]]

   # wk_bad = wk[wk["risky_week_proxy"] == 1]
    # Stable weeks (low visual weight)
    fig3.add_trace(go.Scatter(
        x=wk_ok["load"],
        y=wk_ok["risky_week_proxy"] * 100,  # will be 0 if proxy is 0/1
        mode="markers",
        marker=dict(
            size=12,
            opacity=0.55,
            color="rgba(46,59,143,0.9)",   # muted navy
            line=dict(width=1.5, color="rgba(46,59,143,0.9)")
        ),
        name="Stable weeks",
        hovertemplate="Weekly load: %{x:.0f} min<br>Status: Stable<extra></extra>",
    ))

    # Bad weeks (high visual weight)
    fig3.add_trace(go.Scatter(
        x=wk_bad["load"],
        y=wk_bad["risky_week_proxy"] * 100,  # will be 100 if proxy is 1
        mode="markers",
        marker=dict(
            size=14,
            opacity=0.95,
            color="rgba(217,48,37,0.85)",    # red
            line=dict(width=2.0, color="rgba(120,18,12,0.9)")
        ),
        name="Bad weeks (proxy)",
        hovertemplate="Weekly load: %{x:.0f} min<br>Status: Bad week<extra></extra>",
    ))

    # --- Threshold line + subtle danger-zone shading (no big solid block) ------------
    thr = float(beliefs.risk_threshold)

    # --- 1) Compute safe axis ranges so points aren't clipped ------------------------
    x = wk["load"].to_numpy()
    y = (wk["risky_week_proxy"] * 100).to_numpy()

    # x padding
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    xpad = max(100, 0.08 * (xmax - xmin))  # at least 10 minutes padding
    xrng = [xmin - xpad, xmax + xpad]

    # y padding (so 0 and 100 markers are not cut off)
    yrng = [-5, 105]  # simple and demo-friendly for a probability chart

    # --- 2) Threshold line + danger-zone shading ------------------------------------
    # Light shading to the right of the threshold
    fig3.add_vrect(
        x0=thr,
        x1=xrng[1],
        fillcolor="rgba(217,48,37,0.08)",  # subtle red tint
        line_width=0,
        layer="below"
    )

    # Threshold line
    fig3.add_vline(
        x=thr,
        line_dash="dash",
        line_width=3,
        line_color="rgba(17,24,39,0.75)"
    )

    # Threshold label
    fig3.add_annotation(
        x=thr,
        y=100,
        xref="x",
        yref="y",
        text=f"Threshold ≈ {int(thr)}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(17,24,39,0.75)",
        ax=40,
        ay=-40,
        font=dict(size=24, color="rgba(17,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(17,24,39,0.25)",
        borderwidth=1
    )

    # Danger zone label (optional but nice)
    fig3.add_annotation(
        x=(thr + xrng[1]) / 2,
        y=92,
        xref="x",
        yref="y",
        text="Danger zone",
        showarrow=False,
        font=dict(size=25, color="rgba(217,48,37,0.75)")
    )

    # --- 3) Apply axis ranges + light grid + clean layout ----------------------------
    fig3.update_xaxes(range=xrng, showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig3.update_yaxes(range=yrng, showgrid=True, gridcolor="rgba(17,24,39,0.08)")

    fig3.update_layout(
        xaxis_title="Weekly load (minutes)",
        yaxis_title="Bad-week chance (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=60, b=40),
        legend=dict(
            font=dict(size=23),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=0.4
    ),
    )

    style_plot_axes(fig3)

    st.plotly_chart(fig3, width='stretch',
                    config={
        "displayModeBar": False,   # hides the toolbar entirely
        "scrollZoom": False,
        "doubleClick": "reset",
        "displaylogo": False
    })
   
with right3:
    st.markdown("""
    <div style="
        background-color: #f5e5e4;
        padding: 20px;
        border-radius: 12px;
        border-left: 2px solid #f7463b;
        color: #1f2937;
    ">
    <h4 style="margin:0 0 10px 0; font-size:26px">Belief Card</h4>
      <p style="font-size:23px; margin:0 0 8px 0;"><strong>Risk has a tipping point.</strong></p>
      <p style="font-size:23px; margin:0 0 10px 0;">Past a threshold, bad weeks jump. Guardrails matter.</p>
      <p style="font-size:23px; margin:0;"><strong>Threshold</strong> — where risk starts rising fast</p>
    </div>
    """, unsafe_allow_html=True)


    m1, m2 = st.columns(2, gap="large")

    belief_metric(
        m1,
        title="Threshold (weekly load)",
        value=f"{int(beliefs.risk_threshold)}",
        delta="Beyond this, bad weeks jump",
        delta_color="inverse",
        info_md="""
    **What this means**  
    This is the load where the probability of a “bad weeks” starts rising rapidly (missed sessions and/or high soreness).

    **How it’s used**  
    Scenario Lab avoids plans near/above this threshold when you choose a conservative posture.
    """
    )

    belief_metric(
        m2,
        title="How fast risk rises",
        value=f"{beliefs.risk_slope:.3f}",
        delta="Higher = steeper cliff",
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

   #st.markdown("Implication: optimize for robustness, not just the best average plan.")
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

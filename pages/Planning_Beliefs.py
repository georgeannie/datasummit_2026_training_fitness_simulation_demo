# planning_assumptions.py — Conference-ready, domain-neutral, colored KPIs
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.nav import SCENARIO_PAGE, BELIEFS_PAGE, render_top_nav
from core import (
    get_history_and_beliefs, sat_gain, sigmoid, weighted_load,
    DEFAULT_H, CARRYOVER_MULTIPLIER, BASE_PRIOR
)

# ── Chart sizing for projector in lit room ──
AXIS_TITLE_FONT_SIZE = 28
AXIS_TICK_FONT_SIZE = 22
CHART_HEIGHT = 520
LINE_WIDTH = 6
MARKER_SIZE_SMALL = 16
MARKER_SIZE_LARGE = 18
ANNOTATION_FONT_SIZE = 28

# ── Color per assumption ──
C_BLUE = "#2563EB"
C_PURPLE = "#7C3AED"
C_RED = "#DC2626"
C_GREEN = "#059669"
C_AMBER = "#D97706"

# ── Chart line colors (SWAPPED: interval=blue, easy=red) ──
LINE_INTERVAL = "#0066FF"   # blue — the key lever, flattens first
LINE_EASY = "#E60000"       # red — flattens later
LINE_STRENGTH = "#00994D"   # green


def style_plot_axes(fig):
    fig.update_xaxes(showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                     tickfont=dict(size=AXIS_TICK_FONT_SIZE))
    fig.update_yaxes(showgrid=False, title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                     tickfont=dict(size=AXIS_TICK_FONT_SIZE))


def insight_card(bg, border, label_color, dark_color, plain, business, so_what):
    st.markdown(f"""
    <div style="
        background:{bg}; padding:28px 30px; border-radius:14px;
        border-left:6px solid {border}; color:#111827; margin-bottom:20px;
    ">
      <div style="font-size:16px; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.08em; color:{label_color}; margin-bottom:14px;">
        What this means
      </div>
      <p style="font-size:28px; font-weight:700; line-height:1.35; margin:0 0 18px 0; color:#1A1A1A;">
        {plain}
      </p>
      <div style="font-size:24px; line-height:1.45; color:#4B5563; margin-bottom:18px;
                  padding-bottom:18px; border-bottom:1px solid rgba(0,0,0,0.06);">
        {business}
      </div>
      <div style="font-size:24px; font-weight:700; line-height:1.35; padding:14px 18px;
                  border-radius:10px; background:rgba(0,0,0,0.04); color:{dark_color};">
        💡 {so_what}
      </div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(title, value, delta, delta_direction, delta_color_hex, biz_line):
    """Conference-sized KPI with large colored directional arrow."""
    arrow = "↑" if delta_direction == "up" else "↓"
    st.markdown(f"""
    <div style="
        background:white; border:2px solid #E5E7EB; border-radius:14px;
        padding:22px; min-height:190px;
    ">
      <div style="font-size:20px; color:#6B7280; font-weight:600; margin-bottom:8px;">{title}</div>
      <div style="font-size:52px; font-weight:800; line-height:1.1; color:#1A1A1A;">{value}</div>
      <div style="font-size:24px; font-weight:800; color:{delta_color_hex}; margin-top:10px;">
        <span style="font-size:30px; vertical-align:middle;">{arrow}</span> {delta}
      </div>
      <div style="font-size:18px; color:#9CA3AF; margin-top:10px; font-style:italic; line-height:1.3;">{biz_line}</div>
    </div>
    """, unsafe_allow_html=True)


def assumption_header(number, color, title, oneliner):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:18px; margin:14px 0 14px 0;">
      <div style="
          width:56px; height:56px; border-radius:50%; background:{color};
          display:flex; align-items:center; justify-content:center;
          font-weight:800; font-size:26px; color:white; flex-shrink:0;
      ">{number}</div>
      <div>
        <div style="font-size:36px; font-weight:800; color:#1A1A1A; line-height:1.15;">{title}</div>
        <div style="font-size:24px; color:#6B7280; font-weight:400; margin-top:4px;">{oneliner}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════
st.markdown("""<style>
div[data-testid="stExpander"] {
  background: rgba(255,255,255,0.04); border-radius:14px;
  border:1px solid rgba(255,255,255,0.10);
}
</style>""", unsafe_allow_html=True)

render_top_nav(active=BELIEFS_PAGE)

st.markdown("""
<div style="margin-bottom:12px;">
  <div style="font-size:48px; font-weight:800; color:#1A1A1A; line-height:1.1;">
    What the Model Believes
  </div>

""", unsafe_allow_html=True)
#   <p style="font-size:24px; color:#6B7280; max-width:800px; line-height:1.5; margin-top:12px;">
#     Three assumptions power Scenario Lab. Each is learned from 6 weeks of history.
#   </p>
# </div>

st.markdown(f"""
<div style="
    margin:12px 0 32px 0; padding:18px 22px; background:#EFF6FF;
    border-left:5px solid {C_BLUE}; border-radius:0 10px 10px 0;
    font-size:22px; color:#1E40AF; line-height:1.5;
">
  <strong>6 weeks of data — informed starting points, not certainties.</strong>
  If a curve doesn't match your experience, adjust its shape — then re-evaluate the decision.
</div>
""", unsafe_allow_html=True)

df_daily, df_proxy, beliefs = get_history_and_beliefs(seed=11, weeks=6)
params = beliefs.params


# ═══════════════════════════════════════════
# ASSUMPTION 1 — CARRYOVER
# ═══════════════════════════════════════════
assumption_header(1, C_BLUE, "Effects Persist, Then Fade",
                  "Today's effort helps tomorrow — but not forever")

left2, right2 = st.columns([1.6, 1.0], gap="large")

with left2:
    half_life_days = 7.0 * beliefs.half_life_wks if beliefs.half_life_wks > 0 else 4.0
    days = np.arange(0, 15, 1)
    decay = np.exp(-np.log(2) * (days / max(1e-6, half_life_days)))
    y_day1 = float(np.exp(-np.log(2) * (1 / max(1e-6, half_life_days))))
    y_day7 = float(np.exp(-np.log(2) * (7 / max(1e-6, half_life_days))))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=days, y=decay, mode="lines",
        line=dict(color=C_BLUE, width=LINE_WIDTH),
        hovertemplate="Day %{x}<br>Benefit: %{y:.0%}<extra></extra>",
        showlegend=False))
    fig2.add_vline(x=float(half_life_days), line_color="rgba(17,24,39,0.7)",
                   line_width=5, line_dash="dash")
    fig2.add_annotation(
        x=float(half_life_days), y=0.6,
        text="Half-life", showarrow=True, arrowhead=2, ax=100, ay=-80,
        font=dict(size=ANNOTATION_FONT_SIZE, color="rgba(50,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.92)")
    fig2.add_trace(go.Scatter(
        x=[1, 7], y=[y_day1, y_day7], mode="markers+text",
        marker=dict(size=MARKER_SIZE_LARGE, color=C_BLUE,
                    line=dict(width=2, color="rgba(17,24,39,0.5)")),
        text=[f"Day 1: {y_day1:.0%}", f"Day 7: {y_day7:.0%}"],
        textposition=["top center", "top center"],
        textfont=dict(size=ANNOTATION_FONT_SIZE, color=C_RED),
        showlegend=False))
    fig2.update_layout(
        title=dict(text="Carryover Curve", xanchor="left", x=0, font=dict(size=32)),
        xaxis_title="Days since workout", yaxis_title="Benefit left (%)",
        height=CHART_HEIGHT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=85, b=55), showlegend=False, template="plotly_white")
    fig2.update_yaxes(range=[0, 1.02], tickformat=".0%")
    fig2.update_xaxes(range=[0, 14], showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    style_plot_axes(fig2)
    st.plotly_chart(fig2, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False})

with right2:
    insight_card(
        bg="#EFF6FF", border=C_BLUE, label_color=C_BLUE, dark_color="#1E40AF",
        plain="Benefits fade fast. By day 7, almost nothing remains.",
        business='<strong>In business:</strong> Any investment — a campaign, a restock, a new hire — keeps working after you stop. But the dashboard only measures this week. So slow-build investments look like they\'re not working when they are.',
        so_what="Three consistent weeks beat one heroic week.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        kpi_card("After 7 days", f"{y_day7:.0%}",
                 "Fades quickly", "down", C_RED,
                 "The effect is real — it just shows up later")
    with c2:
        kpi_card("Half-life", f"{float(half_life_days):.1f} days",
                 "Short", "down", C_AMBER,
                 "Spacing matters more than volume")

st.markdown("---")


# ═══════════════════════════════════════════
# ASSUMPTION 2 — DIMINISHING RETURNS
# ═══════════════════════════════════════════
assumption_header(2, C_PURPLE, "More Helps — Until It Doesn't",
                  "Every lever has a flattening point")

left1, right1 = st.columns([1.6, 1.0], gap="large")

with left1:
    x = np.arange(0, 300, 5)
    wk_easy = df_proxy["easy_min"].values
    wk_interval = df_proxy["moderate_run_comfort_pace_min"].values
    wk_strength = df_proxy["strength_min"].values
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=x, y=[sat_gain(m, **params["easy"]) for m in x],
        mode="lines", name="Easy",
        line=dict(color=LINE_EASY, width=LINE_WIDTH)))
    fig1.add_trace(go.Scatter(
        x=x, y=[sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x],
        mode="lines", name="Interval",
        line=dict(color=LINE_INTERVAL, width=LINE_WIDTH)))
    fig1.add_trace(go.Scatter(
        x=x, y=[sat_gain(m, **params["strength"]) for m in x],
        mode="lines", name="Strength",
        line=dict(color=LINE_STRENGTH, width=LINE_WIDTH)))

    for vals, ckey, label in [
        (wk_easy, "rgba(230,0,0,0.5)", "Easy"),
        (wk_interval, "rgba(0,102,255,0.55)", "Interval"),
        (wk_strength, "rgba(0,153,77,0.5)", "Strength")]:
        p_key = {"Easy": "easy", "Interval": "moderate_run_comfort_pace", "Strength": "strength"}[label]
        fig1.add_trace(go.Scatter(
            x=vals, y=[sat_gain(m, **params[p_key]) for m in vals],
            mode="markers",
            marker=dict(size=MARKER_SIZE_SMALL, color=ckey,
                        line=dict(width=2, color="rgba(17,24,39,0.5)")),
            showlegend=False))

    k_interval = float(params["moderate_run_comfort_pace"]["k"])
    fig1.add_vrect(x0=k_interval, x1=max(x), fillcolor="rgba(0,102,255,0.04)",
                   line_width=0, layer="below")
    fig1.add_vline(x=k_interval, line_dash="dash", line_width=4,
                   line_color="rgba(17,24,39,0.6)")
    fig1.add_annotation(
        x=k_interval,
        y=max([sat_gain(m, **params["moderate_run_comfort_pace"]) for m in x]) * 0.96,
        text=f"Flattens ~{int(round(k_interval))} min",
        showarrow=True, arrowhead=2, ax=40, ay=-30,
        font=dict(size=ANNOTATION_FONT_SIZE - 2, color="rgba(17,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(17,24,39,0.20)", borderwidth=1)

    fig1.update_layout(
        title=dict(text="Response Curve", xanchor="left", x=0, font=dict(size=32)),
        xaxis_title="Minutes per week", yaxis_title="Improvement score",
        height=CHART_HEIGHT, margin=dict(l=25, r=20, t=75, b=55),
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=22),
        legend=dict(orientation="v", yanchor="bottom", y=0.70, xanchor="right",
                    font=dict(size=26), x=0.24))
    fig1.update_xaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    style_plot_axes(fig1)
    st.plotly_chart(fig1, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False})

with right1:
    insight_card(
        bg="#F3EEFF", border=C_PURPLE, label_color=C_PURPLE, dark_color="#5B21B6",
        plain="Not all minutes are equal. Interval flattens earliest.",
        business='<strong>In business:</strong> Your top performer — whether it\'s a channel, a product, or a team — may already be on the flat part of the curve. The next unit of investment there barely moves the needle. The underperformer with room to grow gets more from the same resource.',
        so_what='Don\'t add more. Move what you have.')
    params_table = pd.DataFrame([
        {"Lever": "Easy", "Max lift": int(round(float(params["easy"]["alpha"]))),
         "Flattens at": f'{int(round(float(params["easy"]["k"])))} min'},
        {"Lever": "Interval", "Max lift": int(round(float(params["moderate_run_comfort_pace"]["alpha"]))),
         "Flattens at": f'{int(round(float(params["moderate_run_comfort_pace"]["k"])))} min'},
        {"Lever": "Strength", "Max lift": int(round(float(params["strength"]["alpha"]))),
         "Flattens at": f'{int(round(float(params["strength"]["k"])))} min'},
    ])
    st.markdown(" ")
    styled = (params_table.style
        .set_properties(**{"font-size": "24px", "font-weight": "600", "text-align": "left"})
        .set_table_styles([
            {"selector": "table", "props": [("font-size", "24px"), ("border-collapse", "collapse"), ("width", "100%")]},
            {"selector": "th", "props": [("font-size", "24px"), ("font-weight", "800"),
                ("background-color", "#E8E1EE"), ("color", "#5B21B6"), ("padding", "14px 14px"), ("text-align", "right")]},
            {"selector": "td", "props": [("font-size", "24px"), ("font-weight", "600"),
                ("padding", "14px 40px"), ("text-align", "right")]},
            {"selector": "tbody th", "props": [("display", "none")]},
        ]).hide(axis="index"))
    st.markdown(styled.to_html(), unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════
# ASSUMPTION 3 — RISK THRESHOLD
# ═══════════════════════════════════════════
assumption_header(3, C_RED, "Risk Has a Tipping Point",
                  "Past a threshold, bad outcomes spike")

left3, right3 = st.columns([1.6, 1.0], gap="large")

with left3:
    fig3 = go.Figure()
    wk = df_proxy.copy()
    wk["load"] = weighted_load(wk["easy_min"], wk["moderate_run_comfort_pace_min"], wk["strength_min"])
    wk["risky_week_proxy"] = ((wk["avg_soreness"] >= 5.2) | (wk["missed_days"] >= 1)).astype(int)
    wk_bad = wk[wk["risky_week_proxy"].astype(bool)]
    wk_ok = wk[~wk["risky_week_proxy"].astype(bool)]

    fig3.add_trace(go.Scatter(
        x=wk_ok["load"], y=wk_ok["risky_week_proxy"] * 100, mode="markers",
        marker=dict(size=MARKER_SIZE_LARGE, opacity=0.6, color=C_BLUE,
                    line=dict(width=2, color=C_BLUE)), name="Stable weeks"))
    fig3.add_trace(go.Scatter(
        x=wk_bad["load"], y=wk_bad["risky_week_proxy"] * 100, mode="markers",
        marker=dict(size=MARKER_SIZE_LARGE + 2, opacity=0.95,
                    color="rgba(230,0,0,0.85)",
                    line=dict(width=2.5, color="rgba(120,18,12,0.9)")),
        name="Bad weeks"))

    thr = float(beliefs.risk_threshold)
    x_vals = wk["load"].to_numpy()
    xmin, xmax = np.nanmin(x_vals), np.nanmax(x_vals)
    xpad = max(100, 0.08 * (xmax - xmin))
    xrng = [xmin - xpad, xmax + xpad]

    fig3.add_vrect(x0=thr, x1=xrng[1], fillcolor="rgba(230,0,0,0.08)",
                   line_width=0, layer="below")
    fig3.add_vline(x=thr, line_dash="dash", line_width=5,
                   line_color="rgba(17,24,39,0.75)")
    fig3.add_annotation(
        x=thr, y=100, text=f"Threshold ≈ {int(thr)}",
        showarrow=True, arrowhead=2, arrowwidth=3,
        arrowcolor="rgba(17,24,39,0.75)", ax=55, ay=-50,
        font=dict(size=ANNOTATION_FONT_SIZE, color="rgba(17,24,39,0.95)"),
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(17,24,39,0.25)", borderwidth=1)
    fig3.add_annotation(
        x=(thr + xrng[1]) / 2, y=92, text="Danger zone",
        showarrow=False,
        font=dict(size=ANNOTATION_FONT_SIZE + 2, color="rgba(230,0,0,0.7)"))

    fig3.update_xaxes(range=xrng, showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig3.update_yaxes(range=[-5, 105], showgrid=True, gridcolor="rgba(17,24,39,0.08)")
    fig3.update_layout(
        xaxis_title="Weekly load (minutes)", yaxis_title="Bad-week chance (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=CHART_HEIGHT, margin=dict(l=35, r=20, t=65, b=55),
        legend=dict(font=dict(size=26), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=0.4))
    style_plot_axes(fig3)
    st.plotly_chart(fig3, use_container_width=True,
                    config={"displayModeBar": False, "scrollZoom": False, "displaylogo": False})

with right3:
    insight_card(
        bg="#FEF2F2", border=C_RED, label_color=C_RED, dark_color="#991B1B",
        plain="Past a threshold, bad weeks spike — fast.",
        business='<strong>In business:</strong> Any system — a warehouse, a team, a budget — absorbs pressure up to a point. Past that point, one disruption cascades. The failure isn\'t gradual. It\'s sudden.',
        so_what="Best plan isn't highest return. It's the one that holds up.")
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        kpi_card("Threshold", f"{int(beliefs.risk_threshold)} min",
                 "Bad weeks jump past this", "up", C_RED,
                 "The line the system can't cross safely")
    with m2:
        kpi_card("Risk shape", "Steep cliff",
                 "Not a gentle slope", "up", C_RED,
                 "Small push = big damage")

st.markdown("---")

# ── Closing connector ──
st.markdown(f"""
<div style="
    background:#EFF6FF; padding:28px; border-radius:14px;
    border-left:6px solid {C_BLUE}; font-size:24px; line-height:1.6; color:#1f2937;
">
  In marketing, this framework is called <strong>Marketing Mix Modeling (MMM)</strong>.
  But these assumptions — carryover, diminishing returns, and risk thresholds —
  apply anywhere resources are constrained.<br><br>
  These three assumptions power <strong>Scenario Lab</strong>.
  Disagree with a curve? Adjust its shape — then re-evaluate.<br><br>
  <strong>This isn't a forecast. It's a way to make assumptions explicit
  so decisions can be stress-tested under uncertainty.</strong>
</div>
""", unsafe_allow_html=True)

with st.expander("Debug: weekly proxy table"):
    st.dataframe(df_proxy, use_container_width=True)

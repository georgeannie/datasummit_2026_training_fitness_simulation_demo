import streamlit as st
import streamlit.components.v1 as components
from utils.nav import SCENARIO_PAGE, BELIEFS_PAGE, render_top_nav

st.set_page_config(
    page_title="10K Decision Lab",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
<style>
/* ====== Global ====== */
:root {
  --ink: #0b1220;
  --muted: rgba(11,18,32,.70);
  --card: rgba(255,255,255,.82);
  --stroke: rgba(11,18,32,.10);
  --shadow: 0 10px 30px rgba(0,0,0,.08);
  --shadow2: 0 6px 18px rgba(0,0,0,.10);

  --grad1: linear-gradient(120deg, rgba(65,105,225,.18), rgba(255,105,180,.12));
  --grad2: linear-gradient(90deg, rgba(0, 220, 255,.14), rgba(250, 255, 0,.10));
}

html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
/* Force readable ink everywhere (fixes white-on-white) */
html, body, .stApp, [class*="css"], .stMarkdown, .stMarkdown * {
  color: #0b1220 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6,
.stTitle, .stHeader, .stSubheader {
  color: #0b1220 !important;
}

/* Captions / secondary text */
small, .stCaption, .stMarkdown p, .stMarkdown li {
  color: rgba(11,18,32,.78) !important;
}

/* Links */
a, a * { color: #1d4ed8 !important; }

/* Make horizontal rules visible */
hr { border-color: rgba(11,18,32,.12) !important; }
.stApp {
  background:
    radial-gradient(900px 380px at 10% 0%, rgba(65,105,225,.20), transparent 60%),
    radial-gradient(900px 380px at 90% 0%, rgba(255,105,180,.16), transparent 60%),
    radial-gradient(1100px 420px at 50% 100%, rgba(0,220,255,.12), transparent 60%),
    #f7f8fb;
}

/* Hide default Streamlit header/footer (clean demo) */
header[data-testid="stHeader"] { display: none; }
footer { visibility: hidden; }

/* Collapse empty padding above first element */
.block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

[data-testid="stSidebarNav"] { display: none !important; }
/* Sidebar: keep accessible but quiet */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,.55);
  border-right: 1px solid var(--stroke);
  backdrop-filter: blur(10px);
}

/* ====== Top shell card ====== */
.shell {
  border: 1px solid var(--stroke);
  background: var(--card);
  border-radius: 18px;
  box-shadow: var(--shadow);
  padding: 18px 18px 14px 18px;
  position: relative;
  overflow: hidden;
}
.shell:before {
  content: "";
  position: absolute;
  inset: 0;
  background: var(--grad1);
  opacity: .9;
  pointer-events: none;
}
.shell-inner {
  position: relative;
  z-index: 1;
}
.brand {
  display:flex; align-items:center; justify-content:space-between;
  gap: 14px;
}
.brand-left { display:flex; align-items:center; gap: 12px; }
.badge {
  width: 40px; height: 40px; border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  background: rgba(11,18,32,.92);
  box-shadow: var(--shadow2);
}
.badge span { font-size: 20px; }
.hgroup h1 {
  margin: 0; padding: 0;
  font-size: 22px; letter-spacing: -0.3px;
  color: var(--ink);
}
.hgroup p {
  margin: 2px 0 0 0;
  color: var(--muted);
  font-size: 13px;
}

.hr { height: 1px; width: 100%; background: rgba(11,18,32,.10); margin: 14px 0 6px 0; }

/* Buttons */
.stButton>button {
  border-radius: 14px !important;
  padding: 0.62rem 1.0rem !important;
  font-weight: 800 !important;
  border: 1px solid rgba(11,18,32,.14) !important;
  background: rgba(0,0,1,1) !important;
  color: rgba(255,255,255,.95) !important;
}
.stButton>button:hover {
  background: rgba(255,255,255,.95) !important;
  color: rgba(11,18,32,.94) !important;
}

</style>
""",
    unsafe_allow_html=True,
)
# 2) Define pages
home = st.Page("pages/Home.py", title="Home", icon="🏠")
scenario = st.Page("pages/Scenario_Lab.py", title="Scenario Lab", icon="🧪")
beliefs = st.Page("pages/Planning_Beliefs.py", title="Planning Beliefs", icon="🧠")

# 3) Use navigation but hide sidebar nav with config/CSS (you already do CSS)
pg = st.navigation([home, scenario, beliefs], position="sidebar")
pg.run()
# /* ====== Nav chips ====== */
# .navrow { display:flex; align-items:center; gap:10px; flex-wrap: wrap; margin-top: 12px; }
# .chip {
#   border: 1px solid var(--stroke);
#   background: rgba(255,255,255,.70);
#   border-radius: 999px;
#   padding: 8px 12px;
#   display:flex; align-items:center; gap:8px;
#   box-shadow: 0 3px 10px rgba(0,0,0,.05);
# }
# .chip .dot {
#   width: 9px; height: 9px; border-radius: 999px;
#   background: rgba(11,18,32,.25);
# }
# .chip.active {
#   background: rgba(11,18,32,.92);
#   border-color: rgba(11,18,32,.92);
# }
# .chip.active span { color: white; }
# .chip.active .dot { background: rgba(255,255,255,.85); }

# /* ====== Divider ====== */
# .hr {
#   height: 1px; width: 100%;
#   background: rgba(11,18,32,.10);
#   margin: 14px 0 6px 0;
# }

# /* Make Streamlit buttons look sharper */
# .stButton>button {
#   border-radius: 12px;
#   padding: .55rem .9rem;
#   border: 1px solid rgba(11,18,32,.18);
# }
# .stButton>button:hover {
#   border-color: rgba(11,18,32,.32);
#   transform: translateY(-1px);
# }

# /* Page title spacing */
# h2, h3 { letter-spacing: -0.2px; }

# /* --- Button system: clean + modern --- */

# /* Base button */
# .stButton > button {
#   appearance: none;
#   border-radius: 14px !important;
#   padding: 0.62rem 1.0rem !important;
#   font-weight: 800 !important;
#   letter-spacing: -0.1px !important;
#   border: 1px solid rgba(11,18,32,.14) !important;
#   background: rgba(0,0,1,1) !important;
#   color: rgba(255,255,255,.95) !important;
#   box-shadow: 0 8px 18px rgba(0,0,0,.08) !important;
#   transition: transform .12s ease, box-shadow .12s ease, background .12s ease, border-color .12s ease, color .12s ease;
# }

# /* Hover */
# .stButton > button:hover {
#   background: rgba(255,255,255,.95) !important;
#   border-color: rgba(11,18,32,.24) !important;
#   color: rgba(11,18,32,.94) !important;
#   transform: translateY(-1px);
#   box-shadow: 0 10px 22px rgba(0,0,0,.10) !important;
# }

# /* Active (click) */
# .stButton > button:active {
#   transform: translateY(0px);
#   box-shadow: 0 6px 14px rgba(0,0,0,.08) !important;
# }

# /* Focus (keyboard) */
# .stButton > button:focus,
# .stButton > button:focus-visible {
#   outline: none !important;
#   box-shadow: 0 0 0 4px rgba(65,105,225,.18), 0 10px 22px rgba(0,0,0,.10) !important;
#   border-color: rgba(65,105,225,.35) !important;
# }

# /* Ensure Streamlit doesn't recolor text on hover via internal spans */
# .stButton > button * {
#   color: inherit !important;
# }

# /* Disabled */
# .stButton > button:disabled {
#   opacity: .55 !important;
#   box-shadow: none !important;
#   transform: none !important;
# }

# /* --- Optional: primary button class via a wrapper --- */
# /* Use: st.markdown('<div class="primary-btn">', unsafe_allow_html=True); st.button(...); st.markdown('</div>', unsafe_allow_html=True) */
# .primary-btn .stButton > button {
#   background: rgba(11,18,32,.94) !important;
#   color: rgba(255,255,255,.96) !important;
#   border-color: rgba(11,18,32,.94) !important;
#   box-shadow: 0 10px 22px rgba(11,18,32,.18) !important;
# }
# .primary-btn .stButton > button:hover {
#   background: rgba(11,18,32,.98) !important;
#   color: rgba(255,255,255,.98) !important;
#   border-color: rgba(11,18,32,.98) !important;
#   box-shadow: 0 12px 26px rgba(11,18,32,.22) !important;
# }

# /* Secondary subtle button */
# .secondary-btn .stButton > button {
#   background: rgba(255,255,255,.70) !important;
#   color: rgba(11,18,32,.92) !important;
# }

# /* --- Remove any theme-red hover artifacts from links inside buttons (rare but happens) --- */
# .stButton > button a,
# .stButton > button a:hover,
# .stButton > button a:visited {
#   color: inherit !important;
#   text-decoration: none !important;
# }
# -----------------------------
# Helpers
# -----------------------------
# SCENARIO_PAGE = "pages/Scenario_Lab.py"
# BELIEFS_PAGE = "pages/Planning_Beliefs.py"

# # --- Sidebar branding (since default nav is hidden) ---

# def _switch(page_path: str):
#     """
#     Streamlit navigation that stays robust across versions:
#     - Prefer st.switch_page (best UX)
#     - Fall back to st.page_link message (if switch not available)
#     """
#     if hasattr(st, "switch_page"):
#         st.switch_page(page_path)
#     else:
#         st.info("Your Streamlit version doesn’t support programmatic page switching. Use the sidebar navigation.")
#         st.stop()

# def _active_chip(label: str, active: bool) -> str:
#     cls = "chip active" if active else "chip"
#     return f"""
#       <div class="{cls}">
#         <div class="dot"></div>
#         <span>{label}</span>
#       </div>
#     """

# # -----------------------------
# # Read query params for simple state (optional)
# # -----------------------------
# params = st.query_params
# active = params.get("view", "home").lower()
# # -----------------------------
# # Header Shell (persistent)
# # -----------------------------
# components.html(
#       f"""
# <div class="shell">
#   <div class="shell-inner">
#     <div class="brand">
#       <div class="brand-left">
#         <div class="badge"><span>🏁</span></div>
#         <div class="hgroup">
#           <h1>10K Decision Lab</h1>
#           <p>What-if modeling for resource allocation under uncertainty — through training.</p>
#         </div>
#       </div>
#       <div class="chip"><span style="color: rgba(11,18,32,.70);">Demo Mode</span></div>
#     </div>
#     <div class="hr"></div>
#   </div>
# </div>
# """,
#     height=120,
#     scrolling=False,
# )

# #     f"""
# # <div class="shell">
# #   <div class="shell-inner">
# #     <div class="brand">
# #       <div class="brand-left">
# #         <div class="badge"><span>🏁</span></div>
# #         <div class="hgroup">
# #           <h1>10K Decision Lab</h1>
# #           <p>What-if modeling for resource allocation under uncertainty — through training.</p>
# #         </div>
# #       </div>
# #       <div style="display:flex; gap:10px; align-items:center;">
# #         <div class="chip"><span style="color: rgba(11,18,32,.70);">Demo Mode</span></div>
# #       </div>
# #     </div>

# #     <div class="navrow">
# #       {_active_chip("Home", active == "home")}
# #       {_active_chip("Scenario Lab", active == "scenario")}
# #       {_active_chip("Planning Beliefs", active == "beliefs")}
# #     </div>
# #     <div class="hr"></div>
# #   </div>
# # </div>
# # """,
# #     height=140,  # tweak if your shell wraps
# #     scrolling=False,
# # -----------------------------
# # TOP NAV (clickable buttons)
# # -----------------------------
# n1, n2, n3 = st.columns([1, 1, 1], gap="small")
# render_top_nav(active)
# with n1:
#     if st.button("🏠 Home", use_container_width=True, disabled=(active == "home")):
#         st.query_params.update({"view": "home"})
#         # stay on this page (Home)

# with n2:
#     if st.button("Scenario Lab", use_container_width=True, disabled=(active == "scenario")):
#         st.query_params.update({"view": "scenario"})
#         _switch(SCENARIO_PAGE)

# with n3:
#     if st.button("Planning Beliefs", use_container_width=True, disabled=(active == "beliefs")):
#         st.query_params.update({"view": "beliefs"})
#         _switch(BELIEFS_PAGE)

# st.markdown("---")
# # -----------------------------
# # Home content (clean + demo-forward)
# # -----------------------------
# st.write("")  # small breathing room
# st.markdown('<div class="homecard">', unsafe_allow_html=True)
# left, right = st.columns([1.35, 1.0], gap="large")

# with left:
#     st.subheader("How to run this demo")
#     st.markdown(
#         """
# **Start with Scenario Lab.** Make a weekly training decision under uncertainty.  
# Then jump to **Planning Beliefs** to show *why* the recommendation was reasonable.

# **What you will learn:**
# - Why **distributions beat point estimates**
# - How decisions live on a **risk–reward frontier**
# - How “MMM” is really **belief management** (response, carryover, risk)
# """
#     )
#     st.markdown("### Step 1: Make a training decision under uncertainty")
#     st.markdown(
#         """
# 1) “**Constraint:** I have **5 hours/week**.”  
# 2) “**Allocation:** I choose an allocation across **Easy / Tempo / Strength**.”  
# 3) “**Distribution:** I don’t get one answer — I get a **range**.”  
# 4) “**Result:** I pick a plan that is **robust** with guardrails.”  
# 5) “**MMM Assumptions:** We will look at **beliefs** that drive this decision.”
# """
#     )

# with right:
#     st.subheader("Quick Launch")
#     c1, c2 = st.columns(2, gap="small")

#     with c1:
#         if st.button("▶ Start Scenario Lab", use_container_width=True):
#             st.query_params.update({"view": "scenario"})
#             _switch(SCENARIO_PAGE)

#     with c2:
#         if st.button("▶ View Planning Beliefs", use_container_width=True):
#             st.query_params.update({"view": "beliefs"})
#             _switch(BELIEFS_PAGE)
# st.markdown("</div>", unsafe_allow_html=True)
# st.write("")
# colA, colB, colC = st.columns([1.1, 1.2, 1.1], gap="small")

# with colA:
#     st.button("🏠 Home", use_container_width=True, disabled=True)

# with colB:
#     if st.button("Make the Decision → Scenario Lab", use_container_width=True):
#         st.query_params.update({"view": "scenario"})
#         _switch(SCENARIO_PAGE)

# with colC:
#     if st.button("Explain the Decision → Planning Beliefs", use_container_width=True):
#         st.query_params.update({"view": "beliefs"})
#         _switch(BELIEFS_PAGE)
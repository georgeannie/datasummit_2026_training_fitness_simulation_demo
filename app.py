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
beliefs = st.Page("pages/Planning_Beliefs.py", title="Assumptions", icon="🧠")

# 3) Use navigation but hide sidebar nav with config/CSS (you already do CSS)
pg = st.navigation([home, scenario, beliefs], position="sidebar")
pg.run()
# ui/nav.py
import streamlit as st

SCENARIO_PAGE = "pages/Scenario_Lab.py"
BELIEFS_PAGE = "pages/Planning_Beliefs.py"
HOME_PAGE = "pages/Home.py"

def _switch(page_path: str):
    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    st.stop()

def render_top_nav(active: str):
    n1, n2, n3 = st.columns([1, 1, 1], gap="small")
    with n1:
        if st.button("🏠 Home", use_container_width=True, disabled=(active == "home")):
            st.query_params.update({"view": "home"})
            _switch(HOME_PAGE)  # if home is this file; otherwise use your home file path

    with n2:
        if st.button("Scenario Lab", use_container_width=True, disabled=(active == "scenario")):
            st.query_params.update({"view": "scenario"})
            _switch(SCENARIO_PAGE)

    with n3:
        if st.button("Planning Beliefs", use_container_width=True, disabled=(active == "beliefs")):
            st.query_params.update({"view": "beliefs"})
            _switch(BELIEFS_PAGE)

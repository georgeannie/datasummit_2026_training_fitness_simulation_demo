import streamlit as st
from utils.nav import SCENARIO_PAGE, BELIEFS_PAGE, HOME_PAGE, render_top_nav, _switch 
# -----------------------------
# Home content (revised)
# -----------------------------
st.write("")
render_top_nav(active=HOME_PAGE)
left, right = st.columns([1.45, 1.0], gap="large")

with left:
    st.subheader("How to run this demo")
    st.markdown(
        """
**Start with Scenario Lab** to make a weekly training allocation under uncertainty.  
Then jump to **Planning Beliefs** to explain *why* the recommendation was reasonable — and when it would change.
"""
    )

    st.markdown("**What you will learn:**")
    st.markdown(
        """
- Why **distributions beat point estimates**
- How decisions sit on a **risk–reward frontier**
- Why “MMM” is really **belief management** (response, carryover, risk)
"""
    )

# with right:
#     st.subheader("Quick Launch")
#     q1, q2 = st.columns(2, gap="small")
#     with q1:
#         if st.button("▶ Start Scenario Lab", use_container_width=True):
#             st.query_params.update({"view": "scenario"})
#             render_top_nav(SCENARIO_PAGE)
#     with q2:
#         if st.button("▶ View Planning Beliefs", use_container_width=True):
#             st.query_params.update({"view": "beliefs"})
#             render_top_nav(BELIEFS_PAGE)
st.markdown("---")

# NEW: Demo Flow “where step 1 and step 2 happen”
st.subheader("Demo Flow (2 steps)")
f1, f2 = st.columns([1.1, 1.1], gap="large")

with f1:
    st.markdown("### Step 1 — Make the decision")
    st.markdown(
        """
**Page:** Scenario Lab  
**Goal:** Choose an allocation across **Easy / Tempo / Strength** under a time constraint.  
**Output:** A recommended plan + guardrails, based on a **distribution of outcomes**.
"""
    )
    if st.button("Go to Step 1 → Scenario Lab", use_container_width=True):
        st.query_params.update({"view": "scenario"})
        _switch(SCENARIO_PAGE)

with f2:
    st.markdown("### Step 2 — Explain the decision")
    st.markdown(
        """
**Page:** Planning Beliefs  
**Goal:** Show the *3 beliefs* that make the recommendation rational.  
**Beliefs:** **Diminishing returns**, **carryover**, **risk threshold**.  
**Output:** “Why this plan” + “When it would change.”
"""
    )
    if st.button("Go to Step 2 → Planning Beliefs", use_container_width=True):
       # st.query_params.update({"view": "beliefs"})
        _switch(BELIEFS_PAGE)

st.markdown("---")

# Optional: One tight “closing loop” line that sets up your narrative arc
st.markdown(
    """
### Together, it’s a reusable pattern for resource allocation under uncertainty.”
"""
)

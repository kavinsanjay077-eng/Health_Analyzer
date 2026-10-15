"""
Streamlit app: symptom-based health condition analysis
=======================================================
Front end for pipeline.HealthModel. Run with:   streamlit run app.py
"""

import pandas as pd
import streamlit as st

from pipeline import HealthModel
from markov_engine import STATES

st.set_page_config(page_title="Symptom-Based Health Condition Analysis", layout="centered")


# Build the probability tables once, not on every click
@st.cache_resource
def load_model():
    return HealthModel()


model = load_model()


def pretty(symptom):
    return symptom.replace("_", " ")


st.title("Symptom-based health condition analysis")
st.caption(
    "Bayes' theorem ranks the likely conditions; a Markov chain forecasts how severity "
    "may progress. Educational project, not medical advice."
)

# ---------------- input ----------------
selected = st.multiselect(
    "Select the symptoms you have",
    options=model.symptom_list,
    format_func=pretty,
)
n_steps = st.slider("Forecast horizon (time steps ahead)", min_value=1, max_value=10, value=3)

if len(selected) == 0:
    st.info("Pick at least one symptom to begin.")
    st.stop()

if len(selected) < 3:
    st.warning(
        "Fewer than 3 symptoms: the ranking is unreliable. In validation, top-1 accuracy was "
        "about 34% with one symptom and about 90% with three."
    )

if st.button("Analyze", type="primary"):
    result = model.analyze(selected, n_steps=n_steps, top_n=5)

    # ---------------- diagnosis ----------------
    st.subheader("Diagnosis probabilities")
    rank_df = pd.DataFrame(result["ranking"])
    rank_df["Condition"] = rank_df["disease"]
    rank_df["Probability (%)"] = (rank_df["probability"] * 100).round(2)
    rank_df["Starting severity"] = rank_df["initial_state"]
    st.bar_chart(rank_df.set_index("Condition")["Probability (%)"])
    st.dataframe(
        rank_df[["Condition", "Probability (%)", "Starting severity"]],
        hide_index=True,
        width="stretch",
    )

    # ---------------- forecast ----------------
    st.subheader(f"Progression forecast after {n_steps} step(s)")
    fc = pd.Series(result["forecast"]).reindex(STATES) * 100
    cols = st.columns(len(STATES))
    for col, state in zip(cols, STATES):
        col.metric(state, f"{fc[state]:.1f}%")
    st.bar_chart(fc.rename("Probability (%)"))

    st.caption(
        "Forecast = sum over conditions of P(condition | symptoms) x P(state after n steps | condition). "
        "Transition probabilities are modelling assumptions, since the dataset has no time-tracked patients."
    )

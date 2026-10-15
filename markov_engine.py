"""
Markov Chain Disease Progression Model
========================================
Engine 2 of the project. Takes a diagnosed disease (from the Bayes
engine) and forecasts how its severity is likely to evolve over time.

States: Mild -> Moderate -> Severe -> Recovered

IMPORTANT LIMITATION (state this explicitly in the report/viva):
The dataset has no time-tracked patient records, so transition
probabilities cannot be empirically estimated the way the Bayes
priors/likelihoods were. The initial state assignment IS data-driven
(based on the dataset's symptom-severity weights). The transition
matrix itself is a clinically-reasoned, explicitly-labeled assumption,
structurally valid (rows sum to 1) but not fit from longitudinal data.
"""

import numpy as np
import pandas as pd

STATES = ["Mild", "Moderate", "Severe", "Recovered"]


# ---------------------------------------------------------
# STEP 1: Data-driven initial state assignment
# ---------------------------------------------------------
def compute_disease_severity(df, symptom_cols, severity_csv_path):
    sev = pd.read_csv(severity_csv_path)
    sev["Symptom"] = sev["Symptom"].str.strip().str.lower().str.replace(" ", "_")
    weight_map = dict(zip(sev["Symptom"], sev["weight"]))

    def row_avg_score(row):
        weights = [weight_map[row[c]] for c in symptom_cols
                   if row[c] is not None and row[c] in weight_map]
        return sum(weights) / len(weights) if weights else 0

    df = df.copy()
    df["severity_avg"] = df.apply(row_avg_score, axis=1)

    disease_severity = df.groupby("Disease")["severity_avg"].mean()

    # Thresholds from the data itself (33rd / 66th percentile), not guessed
    low_cut = df["severity_avg"].quantile(0.33)
    high_cut = df["severity_avg"].quantile(0.66)

    return disease_severity, low_cut, high_cut


def initial_state(disease, disease_severity, low_cut, high_cut):
    score = disease_severity[disease]
    if score <= low_cut:
        return "Mild"
    elif score <= high_cut:
        return "Moderate"
    else:
        return "Severe"


# ---------------------------------------------------------
# STEP 2: Transition matrix (explicitly-labeled assumption)
# ---------------------------------------------------------
# Rows = current state, columns = next state. Order matches STATES.
# Reasoning: the chance of recovering in one step FALLS as severity rises
# (Mild 0.23 > Moderate 0.15 > Severe 0.10), and severe cases are the
# stickiest (0.70 chance of staying severe). Recovered is treated as
# near-absorbing with a small relapse chance back to Mild.
TRANSITION_MATRIX = np.array([
    # Mild        Moderate   Severe   Recovered
    [0.55,        0.20,      0.02,    0.23],   # from Mild
    [0.15,        0.45,      0.25,    0.15],   # from Moderate
    [0.00,        0.20,      0.70,    0.10],   # from Severe
    [0.05,        0.00,      0.00,    0.95],   # from Recovered
])

# Sanity check baked in: every row must sum to 1 (stochastic matrix)
assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0), "Transition matrix rows must sum to 1!"


# ---------------------------------------------------------
# STEP 3: Multi-step forecasting
# ---------------------------------------------------------
def forecast(state_name, n_steps):
    """Return the probability distribution over states after n_steps,
    starting from a known current state."""
    state_vector = np.zeros(len(STATES))
    state_vector[STATES.index(state_name)] = 1.0

    # n-step transition = state_vector @ (TRANSITION_MATRIX ** n)
    # via repeated matrix multiplication (Chapman-Kolmogorov)
    matrix_power = np.linalg.matrix_power(TRANSITION_MATRIX, n_steps)
    result = state_vector @ matrix_power
    return dict(zip(STATES, result))


# ---------------------------------------------------------
# STEP 4: Run it
# ---------------------------------------------------------
if __name__ == "__main__":
    from bayes_engine import load_and_clean

    df, symptom_cols = load_and_clean("dataset.csv")
    disease_severity, low_cut, high_cut = compute_disease_severity(
        df, symptom_cols, "Symptom-severity.csv"
    )

    print(f"Severity thresholds -> Mild: <= {low_cut:.2f}, Moderate: <= {high_cut:.2f}, else Severe\n")

    test_disease = "Fungal infection"
    start = initial_state(test_disease, disease_severity, low_cut, high_cut)
    print(f"{test_disease} -> initial state: {start}\n")

    for n in [1, 2, 5]:
        dist = forecast(start, n)
        print(f"After {n} step(s):")
        for s, p in dist.items():
            print(f"  {s:12s} {p*100:.2f}%")
        print()

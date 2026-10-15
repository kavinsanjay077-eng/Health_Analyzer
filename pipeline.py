"""
Pipeline: symptoms -> diagnosis (Bayes) -> progression forecast (Markov)
==========================================================================
Joins Engine 1 and Engine 2 into one function the app can call.

The link between the engines is the law of total probability (Module 1):

    P(state after n steps | symptoms)
        = sum over diseases d of  P(d | symptoms) * P(state after n steps | d)

P(d | symptoms) comes from the Bayes engine. P(state after n steps | d)
comes from the Markov engine, starting in disease d's initial severity state.
So the forecast is weighted by how likely each disease is, instead of
betting everything on the single top diagnosis.
"""

import numpy as np

from bayes_engine import load_and_clean, build_probability_tables, predict
from markov_engine import (
    STATES,
    compute_disease_severity,
    initial_state,
    forecast,
)


class HealthModel:
    def __init__(self, dataset_path="dataset.csv", severity_path="Symptom-severity.csv"):
        df, symptom_cols = load_and_clean(dataset_path)
        self.priors, self.likelihoods, self.all_symptoms = build_probability_tables(df, symptom_cols)
        self.disease_severity, self.low_cut, self.high_cut = compute_disease_severity(
            df, symptom_cols, severity_path
        )
        self.symptom_list = sorted(self.all_symptoms)

    def analyze(self, symptoms, n_steps=3, top_n=5):
        """Return the ranked diagnosis and the progression forecast."""
        # Posterior over ALL diseases (top_n = number of diseases)
        full = predict(symptoms, self.priors, self.likelihoods, self.all_symptoms,
                       top_n=len(self.priors))

        # Ranked list with each disease's data-driven starting severity
        ranking = [
            {
                "disease": d,
                "probability": p,
                "initial_state": initial_state(d, self.disease_severity, self.low_cut, self.high_cut),
            }
            for d, p in full[:top_n]
        ]

        # Posterior-weighted forecast over ALL diseases
        mixed = np.zeros(len(STATES))
        for d, p in full:
            start = initial_state(d, self.disease_severity, self.low_cut, self.high_cut)
            dist = forecast(start, n_steps)
            mixed += p * np.array([dist[s] for s in STATES])

        return {
            "ranking": ranking,
            "forecast": dict(zip(STATES, mixed)),
            "n_steps": n_steps,
        }


if __name__ == "__main__":
    model = HealthModel()

    for symptoms in (
        ["itching", "skin_rash", "nodal_skin_eruptions"],
        ["fatigue", "vomiting", "high_fever"],
    ):
        out = model.analyze(symptoms, n_steps=3)
        print(f"Symptoms: {symptoms}\n")
        print("Diagnosis:")
        for r in out["ranking"]:
            print(f"  {r['disease']:28s} {r['probability'] * 100:6.2f}%   starts: {r['initial_state']}")
        print(f"\nForecast after {out['n_steps']} steps (posterior-weighted):")
        for s, p in out["forecast"].items():
            print(f"  {s:10s} {p * 100:6.2f}%")
        print("\n" + "-" * 50 + "\n")

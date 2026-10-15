
import pandas as pd


# ---------------------------------------------------------
# STEP 1: Load and clean the raw data
# ---------------------------------------------------------
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    symptom_cols = [c for c in df.columns if c.startswith("Symptom")]

    # For every symptom cell: strip whitespace, lowercase, fix stray
    # inner spaces (e.g. "dischromic _patches" -> "dischromic_patches")
    def clean_cell(cell):
        if pd.isna(cell):
            return None
        return cell.strip().lower().replace(" ", "_").replace("__", "_")

    for col in symptom_cols:
        df[col] = df[col].apply(clean_cell)

    return df, symptom_cols


# ---------------------------------------------------------
# STEP 2: Build the probability tables ("training")
# ---------------------------------------------------------
def build_probability_tables(df, symptom_cols):
    diseases = df["Disease"].unique()
    total_rows = len(df)

    # P(Disease) -- the prior. How common is each disease in the dataset.
    priors = {}
    for d in diseases:
        priors[d] = (df["Disease"] == d).sum() / total_rows

    # Collect the full symptom vocabulary (needed for smoothing)
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(df[col].dropna().unique())

    # P(symptom | Disease) -- the likelihood table.
    # likelihoods[disease][symptom] = probability that a patient with
    # this disease exhibits this symptom.
    likelihoods = {}
    for d in diseases:
        disease_rows = df[df["Disease"] == d]
        n_rows = len(disease_rows)

        # Count how many rows (for this disease) contain each symptom
        symptom_counts = {}
        for col in symptom_cols:
            for val in disease_rows[col]:
                if val is not None:
                    symptom_counts[val] = symptom_counts.get(val, 0) + 1

        # Laplace smoothing: add 1 to every count, add len(all_symptoms)
        # to the denominator. This prevents a probability of exactly 0
        # for a symptom that never appeared for this disease in the
        
        likelihoods[d] = {}
        for s in all_symptoms:
            count = symptom_counts.get(s, 0)
            likelihoods[d][s] = (count + 1) / (n_rows + len(all_symptoms))

    return priors, likelihoods, all_symptoms


# ---------------------------------------------------------
# STEP 3: Inference -- given new symptoms, rank diseases
# ---------------------------------------------------------
def predict(symptoms_input, priors, likelihoods, all_symptoms, top_n=5):
    cleaned_input = [s.strip().lower().replace(" ", "_") for s in symptoms_input]

    unknown = [s for s in cleaned_input if s not in all_symptoms]
    if unknown:
        print(f"Warning: these symptoms aren't in the dataset and will be ignored: {unknown}")
    cleaned_input = [s for s in cleaned_input if s in all_symptoms]

    scores = {}
    for d in priors:
        score = priors[d]
        for s in cleaned_input:
            score *= likelihoods[d][s]
        scores[d] = score

    
    total = sum(scores.values())
    ranked = sorted(
        ((d, s / total) for d, s in scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


# ---------------------------------------------------------
# STEP 4: Run
# ---------------------------------------------------------
if __name__ == "__main__":
    df, symptom_cols = load_and_clean("dataset.csv")
    priors, likelihoods, all_symptoms = build_probability_tables(df, symptom_cols)

    print(f"Loaded {len(df)} rows, {len(priors)} diseases, {len(all_symptoms)} unique symptoms.\n")

    test_symptoms = ["cough", "vomiting", "spotting_ urination"]
    print(f"Input symptoms: {test_symptoms}\n")

    results = predict(test_symptoms, priors, likelihoods, all_symptoms)
    print("Ranked predictions:")
    for disease, prob in results:
        print(f"  {disease:30s}  {prob*100:.2f}%")

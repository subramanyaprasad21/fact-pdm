"""
Validation experiments supporting Section 7 of the paper.

Three controlled experiments that strengthen FACT beyond the development data:
  1. external_validation  - run the pipeline on datasets not used to build FACT
  2. permutation_control  - shuffle labels; the feasibility lift must collapse to ~1
  3. model_family_check   - repeat key comparisons with a neural network

Author: Subramanya Prasad Parashuram
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score


def feasibility_lift(X, y, seed=42):
    """PR-AUC lift over a prevalence baseline under 5-fold CV."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(
        RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        X, y, cv=skf, method="predict_proba")[:, 1]
    return average_precision_score(y, proba) / np.mean(y)


def permutation_control(X, y, n_runs=3, seed=0):
    """Negative control: shuffle labels, signal is destroyed, lift must drop to ~1.
    Returns (true_lift, mean_permuted_lift)."""
    true = feasibility_lift(X, y)
    rng = np.random.RandomState(seed)
    permuted = [feasibility_lift(X, rng.permutation(y), seed=s) for s in range(1, n_runs + 1)]
    return round(true, 2), round(float(np.mean(permuted)), 2)


def mlp_baseline_vs_smote(X, y):
    """Repeat the baseline-vs-SMOTE comparison with a multilayer perceptron
    to confirm findings are not specific to tree ensembles."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import make_scorer, f1_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = {"f1": make_scorer(f1_score, zero_division=0)}
    base = ImbPipeline([("sc", StandardScaler()),
                        ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))])
    smote = ImbPipeline([("sc", StandardScaler()), ("s", SMOTE(random_state=42)),
                         ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))])
    b = cross_validate(base, X, y, cv=skf, scoring=f1, n_jobs=-1)["test_f1"].mean()
    s = cross_validate(smote, X, y, cv=skf, scoring=f1, n_jobs=-1)["test_f1"].mean()
    return {"baseline_f1": round(float(b), 3), "smote_f1": round(float(s), 3)}


if __name__ == "__main__":
    print("Import feasibility_lift, permutation_control, or mlp_baseline_vs_smote.")
    print("See validation/README.md for the full results and data sources.")

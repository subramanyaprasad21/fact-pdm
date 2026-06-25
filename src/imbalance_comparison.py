"""
FACT Stage 2: Imbalance Assessment
Compares four imbalance-handling strategies using 5-fold stratified
cross-validation with paired t-tests, reporting precision, recall, and F1
rather than accuracy alone.

Strategies: baseline, class weights, SMOTE, random undersampling.

Author: Subramanya Prasad Parashuram
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats

RANDOM_STATE = 42
N_FOLDS = 5


def compare_strategies(X, y):
    """Run 5-fold CV across four strategies; return metrics and per-fold F1."""
    avg = "binary" if pd.Series(y).nunique() == 2 else "weighted"
    scoring = {
        "precision": make_scorer(precision_score, average=avg, zero_division=0),
        "recall": make_scorer(recall_score, average=avg, zero_division=0),
        "f1": make_scorer(f1_score, average=avg, zero_division=0),
    }
    min_class = pd.Series(y).value_counts().min()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    strategies = {
        "Baseline": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "ClassWeights": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                               class_weight="balanced", n_jobs=-1),
        "SMOTE": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=min(5, max(1, min_class - 1)))),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
        "Undersample": ImbPipeline([
            ("us", RandomUnderSampler(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))]),
    }

    out, f1_folds = {}, {}
    for name, model in strategies.items():
        cv = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        out[name] = {m: f"{cv['test_' + m].mean():.3f} ± {cv['test_' + m].std():.3f}"
                     for m in ["precision", "recall", "f1"]}
        f1_folds[name] = cv["test_f1"]

    # Paired t-test: baseline vs SMOTE
    if "Baseline" in f1_folds and "SMOTE" in f1_folds:
        t, p = stats.ttest_rel(f1_folds["Baseline"], f1_folds["SMOTE"])
        out["significance_baseline_vs_smote"] = {
            "p_value": round(float(p), 3),
            "significant_at_0.05": bool(p < 0.05)}
    return out


if __name__ == "__main__":
    print("Import compare_strategies(X, y) and pass a prepared feature matrix and target.")

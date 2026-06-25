"""
FACT Stage 1: Feasibility Screening
Measures whether a dataset can support predictive maintenance by computing
the lift of model precision-recall performance over a random baseline.

Decision rule:
    lift > 2.0   -> proceed
    1.2 - 2.0    -> proceed with caution
    lift < 1.2   -> stop, address data quality before modelling

Author: Subramanya Prasad Parashuram
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, accuracy_score


def feasibility_lift(X, y, random_state=42):
    """Return PR-AUC lift over random for binary, or accuracy lift for multiclass."""
    n_classes = pd.Series(y).nunique()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    if n_classes == 2:
        proba = clf.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, proba)
        baseline = np.mean(y_test)
        return {"metric": "PR-AUC", "score": round(pr_auc, 3),
                "random_baseline": round(baseline, 3),
                "lift": round(pr_auc / baseline, 2)}
    else:
        acc = accuracy_score(y_test, clf.predict(X_test))
        baseline = 1.0 / n_classes
        return {"metric": "accuracy", "score": round(acc, 3),
                "random_baseline": round(baseline, 3),
                "lift": round(acc / baseline, 2)}


def verdict(lift):
    if lift > 2.0:
        return "PROCEED - learnable signal present"
    if lift >= 1.2:
        return "CAUTION - weak signal, proceed carefully"
    return "STOP - no learnable signal, fix data collection first"


if __name__ == "__main__":
    # Example usage with a generic CSV. Adapt feature/target columns per dataset.
    # df = pd.read_csv("data/cnc/predictive_maintenance.csv")
    # X = df[feature_cols]; y = df[target_col]
    # result = feasibility_lift(X, y)
    # print(result, verdict(result["lift"]))
    print("Import feasibility_lift and verdict, or adapt the example block.")

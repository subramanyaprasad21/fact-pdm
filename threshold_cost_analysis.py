"""
FACT Stage 3: Cost-Tuned Deployment
Two analyses:
  1. Threshold tuning on the baseline model as an alternative to resampling.
  2. Cost-sensitive optimal threshold selection given an FN:FP cost ratio.

Author: Subramanya Prasad Parashuram
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42


def threshold_sweep(X, y, thresholds=(0.5, 0.4, 0.3, 0.2, 0.1)):
    """Train baseline RF, sweep decision thresholds, report P/R/F1."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    rows = {}
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows[t] = {
            "precision": round(precision_score(y_test, pred, zero_division=0), 3),
            "recall": round(recall_score(y_test, pred, zero_division=0), 3),
            "f1": round(f1_score(y_test, pred, zero_division=0), 3)}
    return rows


def optimal_threshold_by_cost(X, y, cost_ratios=(1, 5, 10, 50, 100)):
    """Find the threshold minimising total cost for each FN:FP cost ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    out = {}
    for ratio in cost_ratios:
        best_cost, best_t = float("inf"), 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            pred = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            cost = fn * ratio + fp * 1
            if cost < best_cost:
                best_cost, best_t = cost, round(float(t), 2)
        out[f"{ratio}:1"] = best_t
    return out


if __name__ == "__main__":
    print("Import threshold_sweep and optimal_threshold_by_cost; pass prepared X, y.")

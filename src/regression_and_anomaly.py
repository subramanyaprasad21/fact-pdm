"""
FACT applied to non-binary tasks: regression and anomaly detection.

Two predictive maintenance paradigms beyond binary classification:
  1. Regression  - predict a continuous degradation coefficient (gas turbine).
  2. Anomaly detection - flag abnormal readings when no failure labels exist
     (air compressor). Feasibility here is judged by whether anomalies
     concentrate on physically meaningful sensors rather than scattering.

Author: Subramanya Prasad Parashuram
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


def regression_feasibility(X, y, random_state=42):
    """5-fold CV regression. High R2 with low MAE indicates a learnable
    degradation signal suitable for predictive maintenance."""
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {"r2": make_scorer(r2_score),
               "mae": make_scorer(mean_absolute_error)}
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    cv = cross_validate(rf, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return {"r2_mean": round(float(cv["test_r2"].mean()), 4),
            "r2_std": round(float(cv["test_r2"].std()), 4),
            "mae_mean": round(float(cv["test_mae"].mean()), 5)}


def anomaly_detection(X, sensors, contamination=0.05, random_state=42):
    """Unsupervised anomaly detection when no failure labels exist.
    Returns the anomaly rate and the sensors that most distinguish
    anomalous from normal readings (physical plausibility check)."""
    Xc = X[sensors].dropna()
    Xs = StandardScaler().fit_transform(Xc)
    iso = IsolationForest(contamination=contamination,
                          random_state=random_state, n_jobs=-1)
    labels = iso.fit_predict(Xs)
    anom, norm = Xc[labels == -1], Xc[labels == 1]
    diffs = [(s, abs(anom[s].mean() - norm[s].mean()) / (norm[s].std() + 1e-9))
             for s in sensors]
    diffs.sort(key=lambda t: -t[1])
    return {"anomaly_rate": round(float((labels == -1).mean()), 3),
            "top_anomaly_sensors": [s for s, _ in diffs[:3]]}


if __name__ == "__main__":
    print("Import regression_feasibility(X, y) or anomaly_detection(X, sensors).")

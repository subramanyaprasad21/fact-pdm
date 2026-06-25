# External Validation

This directory documents the validation of FACT on datasets and experiments that
were **not** used to develop the framework. It supports the claims in Section 7
of the paper.

## Independent datasets (Step 1 and Step 2 validation)

Six datasets across four industries not present in the original six:

| Dataset | Industry | Imbalance | Feasibility lift | Baseline F1 | SMOTE F1 | Note |
|---------|----------|-----------|------------------|-------------|----------|------|
| C-MAPSS FD001 | Aerospace turbofan | 5.7:1 | 6.3x | 0.863 | 0.852 | baseline > SMOTE (p=0.003) |
| C-MAPSS FD002 | Aerospace turbofan | 5.7:1 | 6.2x | 0.849 | 0.844 | 6 operating conditions |
| C-MAPSS FD003 | Aerospace turbofan | 7.0:1 | 7.6x | 0.871 | 0.862 | 2 fault modes |
| Gearbox fault | Rotating machinery | 1.0:1 | 2.0x | 0.983 | 0.983 | balanced; identical |
| Pump sensors | Water infrastructure | 10:1 | 11.0x | 0.997 | 0.997 | strong signal |
| SECOM | Semiconductor mfg | 14:1 | 2.1x | 0.000 | 0.156 | HARD: baseline fails, resampling essential (p=0.004) |

Two findings held across all six:
1. **Feasibility screen (Step 1)** correctly flagged every learnable dataset (lift > 1.2x).
2. **Imbalance assessment (Step 2)** confirmed that strategy value scales with difficulty:
   on easy/moderate data baseline >= SMOTE; on hard SECOM resampling is essential.
   Significant effects in BOTH directions (p=0.003 and p=0.004) on independent data.

## Controlled experiments

### Label-permutation negative control (feasibility screen)
Shuffling labels destroys signal by construction; a sound feasibility screen must
collapse toward lift = 1.0.

| Dataset | True lift | Permuted lift |
|---------|-----------|---------------|
| CNC | 23.3x | 1.07x |
| Gas turbine | 6.4x | 0.99x |

The screen returns to ~1.0 when signal is removed, confirming it responds to
learnable signal rather than incidental dataset properties.

### Model-family robustness (MLP vs random forest)
The key comparisons were repeated with a multilayer perceptron:

| Dataset | Model | Baseline F1 | SMOTE F1 |
|---------|-------|-------------|----------|
| CNC | MLP | 0.723 | 0.607 |
| SECOM | MLP | 0.128 | 0.131 |

Same patterns as the tree ensembles: findings are about the data, not the model family.

### Cost-tuned threshold savings (Step 3 demonstration)
Cost-optimal threshold vs FN:FP cost ratio on out-of-fold CNC predictions:

| FN:FP | Optimal threshold | Saving vs default 0.5 |
|-------|-------------------|------------------------|
| 1:1 | 0.40 | 8% |
| 10:1 | 0.18 | 46% |
| 100:1 | 0.02 | 78% |

The optimal threshold falls and savings grow as missed-failure cost rises.

## Result files
See the JSON files in `../results/` prefixed `validation_` and `cost_savings_curve.json`.

## Data sources
- C-MAPSS: NASA Prognostics Center of Excellence (public domain)
- Gearbox Fault Diagnosis: Kaggle (brjapon/gearbox-fault-diagnosis)
- Pump sensor data: Kaggle (nphantawee/pump-sensor-data)
- SECOM: UCI Machine Learning Repository
Datasets are not redistributed here; download from the original sources.

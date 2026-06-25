# Datasets

This study uses six publicly available industrial datasets spanning three
predictive maintenance paradigms (classification, regression, anomaly
detection). They are NOT redistributed here. Download each from its original
source and place the files in this folder.

| Dataset | Task | Source |
|---------|------|--------|
| CNC machining (AI4I-style milling) | Classification | UCI / Kaggle |
| Irrigation machine | Classification | Kaggle |
| EV charging stations | Classification | Kaggle |
| Microsoft Azure PdM (industrial equipment) | Classification | Microsoft / Kaggle Azure Predictive Maintenance |
| Marine gas turbine (CBM) | Regression | UCI Machine Learning Repository |
| MetroPT-3 air compressor | Anomaly detection | UCI Machine Learning Repository |

Expected layout:

```
data/
  cnc/predictive_maintenance.csv
  irrigation/irrigation_machine.csv
  ev/detailed_ev_charging_stations.csv
  azure/PdM_telemetry.csv, PdM_failures.csv, PdM_machines.csv
  gas_turbine/data.csv
  air_compressor/MetroPT3.csv
```

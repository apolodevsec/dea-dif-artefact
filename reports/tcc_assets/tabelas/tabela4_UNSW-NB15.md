**Tabela 4 — UNSW-NB15**

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1 (%) | AUC-ROC | FPR (%) | Latência (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 — iForest (features brutas) | 67.40 | 64.66 | 63.92 | 64.29 | 0.6949 | 29.65 | 0.0084 |
| B2 — FC-DAE isolado (MSE) | 68.28 | 64.95 | 67.13 | 66.02 | 0.7196 | 30.75 | 0.0000 |
| B3 — Pre-IF + iForest clássico | 68.35 | 65.06 | 67.08 | 66.05 | 0.7473 | 30.58 | 0.0055 |
| B3b — Pre-IF + DIF puro (λ = 0) | 68.16 | 60.60 | 87.59 | 71.64 | 0.7737 | 48.33 | 0.0744 |
| B4 — Random Forest supervisionado | 91.81 | 92.02 | 89.97 | 90.98 | 0.9807 | 6.63 | 0.0034 |
| B6a — Fusão linear ponderada | 72.13 | 70.41 | 67.79 | 69.08 | 0.7712 | 24.18 | 0.0744 |
| B6b — Regra disjuntiva (OR) | 62.64 | 55.88 | 88.44 | 68.49 | 0.7623 | 59.27 | 0.0744 |

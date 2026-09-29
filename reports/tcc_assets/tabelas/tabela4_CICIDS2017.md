**Tabela 4 — CICIDS2017**

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1 (%) | AUC-ROC | FPR (%) | Latência (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 — iForest (features brutas) | 89.34 | 62.24 | 74.35 | 67.76 | 0.8768 | 8.00 | 0.0045 |
| B2 — FC-DAE isolado (MSE) | 89.66 | 62.48 | 78.56 | 69.60 | 0.9051 | 8.37 | 0.0000 |
| B3 — Pre-IF + iForest clássico | 71.16 | 32.91 | 87.99 | 47.90 | 0.8482 | 31.83 | 0.0049 |
| B3b — Pre-IF + DIF puro (λ = 0) | 72.33 | 33.38 | 83.99 | 47.77 | 0.8533 | 29.74 | 0.0455 |
| B4 — Random Forest supervisionado | 99.85 | 99.70 | 99.33 | 99.51 | 0.9998 | 0.05 | 0.0014 |
| B6a — Fusão linear ponderada | 89.66 | 62.48 | 78.56 | 69.60 | 0.9036 | 8.37 | 0.0455 |
| B6b — Regra disjuntiva (OR) | 73.66 | 35.05 | 87.76 | 50.10 | 0.8794 | 28.85 | 0.0455 |

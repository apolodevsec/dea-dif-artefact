**Tabela 4 — NSL-KDD**

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1 (%) | AUC-ROC | FPR (%) | Latência (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 — iForest (features brutas) | 90.42 | 92.76 | 86.80 | 89.68 | 0.9701 | 6.24 | 0.0057 |
| B2 — FC-DAE isolado (MSE) | 91.54 | 91.09 | 91.29 | 91.19 | 0.9765 | 8.22 | 0.0000 |
| B3 — Pre-IF + iForest clássico | 85.14 | 80.89 | 90.37 | 85.37 | 0.9094 | 19.67 | 0.0054 |
| B3b — Pre-IF + DIF puro (λ = 0) | 85.69 | 80.15 | 93.25 | 86.20 | 0.9021 | 21.27 | 0.0792 |
| B4 — Random Forest supervisionado | 99.56 | 99.72 | 99.36 | 99.54 | 0.9998 | 0.26 | 0.0025 |
| B6a — Fusão linear ponderada | 92.67 | 91.74 | 93.09 | 92.41 | 0.9703 | 7.72 | 0.0792 |
| B6b — Regra disjuntiva (OR) | 85.97 | 78.74 | 96.91 | 86.88 | 0.9669 | 24.11 | 0.0792 |

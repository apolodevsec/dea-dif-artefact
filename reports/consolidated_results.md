# Consolidação Multi-Dataset — FC-DAE + Deep Isolation Forest

Execução do mesmo protocolo *leakage-free* sobre cada dataset processado. Todas as métricas referem-se ao conjunto de teste cego.

## Visão Geral das Execuções

| Dataset | n features | m latente | Treino benigno | Teste | MSE val. do FC-DAE |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **CICIDS2017** | 77 | 9 | 1,128,520 | 669,950 | 0.000214 |
| **NSL-KDD** | 115 | 14 | 45,795 | 44,360 | 0.003684 |
| **UNSW-NB15** | 194 | 24 | 51,004 | 47,545 | 0.003292 |

## Arquitetura Proposta (B6a — Fusão Linear Ponderada)

| Dataset | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | FPR | alfa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CICIDS2017** | 89.66% | 62.48% | 78.56% | 69.60% | 0.9036 | 8.37% | 0.00 |
| **NSL-KDD** | 92.67% | 91.74% | 93.09% | 92.41% | 0.9703 | 7.72% | 0.72 |
| **UNSW-NB15** | 72.13% | 70.41% | 67.79% | 69.08% | 0.7712 | 24.18% | 0.77 |

## Ablação do DEAS por Dataset

| Dataset | F1 B3b (lambda=0) | F1 B6a (DEAS) | Delta F1 | AUC B3b | AUC B6a | Delta AUC | alfa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CICIDS2017** ⚠ | 47.77% | 69.60% | `+0.2183` | 0.8533 | 0.9036 | `+0.0503` | 0.00 |
| **NSL-KDD** | 86.20% | 92.41% | `+0.0621` | 0.9021 | 0.9703 | `+0.0682` | 0.72 |
| **UNSW-NB15** | 71.64% | 69.08% | `-0.0256` | 0.7737 | 0.7712 | `-0.0025` | 0.77 |

> [!CAUTION]
> **⚠ Fusão degenerada em CICIDS2017.** Nesses datasets a calibração levou `alfa` a um extremo do intervalo, de modo que B6a colapsa num de seus dois termos e o `Delta` da linha **não** mede a contribuição do DEAS. Detalhes no relatório de cada dataset.

## Tabela 4 Completa por Dataset

### CICIDS2017

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 89.34% | 62.24% | 74.35% | 67.76% | 0.8768 | 8.00% | 0.0045 |
| **B2_Autoencoder_Alone** | 89.66% | 62.48% | 78.56% | 69.60% | 0.9051 | 8.37% | 0.0000 |
| **B3_PreIF_iForest** | 71.16% | 32.91% | 87.99% | 47.90% | 0.8482 | 31.83% | 0.0049 |
| **B3b_PreIF_DIF_Pure** | 72.33% | 33.38% | 83.99% | 47.77% | 0.8533 | 29.74% | 0.0455 |
| **B4_RandomForest_Supervised** | 99.85% | 99.70% | 99.33% | 99.51% | 0.9998 | 0.05% | 0.0014 |
| **B6a_Hybrid_Linear** | 89.66% | 62.48% | 78.56% | 69.60% | 0.9036 | 8.37% | 0.0455 |
| **B6b_Hybrid_OR** | 73.66% | 35.05% | 87.76% | 50.10% | 0.8794 | 28.85% | 0.0455 |

### NSL-KDD

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 90.42% | 92.76% | 86.80% | 89.68% | 0.9701 | 6.24% | 0.0057 |
| **B2_Autoencoder_Alone** | 91.54% | 91.09% | 91.29% | 91.19% | 0.9765 | 8.22% | 0.0000 |
| **B3_PreIF_iForest** | 85.14% | 80.89% | 90.37% | 85.37% | 0.9094 | 19.67% | 0.0054 |
| **B3b_PreIF_DIF_Pure** | 85.69% | 80.15% | 93.25% | 86.20% | 0.9021 | 21.27% | 0.0792 |
| **B4_RandomForest_Supervised** | 99.56% | 99.72% | 99.36% | 99.54% | 0.9998 | 0.26% | 0.0025 |
| **B6a_Hybrid_Linear** | 92.67% | 91.74% | 93.09% | 92.41% | 0.9703 | 7.72% | 0.0792 |
| **B6b_Hybrid_OR** | 85.97% | 78.74% | 96.91% | 86.88% | 0.9669 | 24.11% | 0.0792 |

### UNSW-NB15

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 67.40% | 64.66% | 63.92% | 64.29% | 0.6949 | 29.65% | 0.0084 |
| **B2_Autoencoder_Alone** | 68.28% | 64.95% | 67.13% | 66.02% | 0.7196 | 30.75% | 0.0000 |
| **B3_PreIF_iForest** | 68.35% | 65.06% | 67.08% | 66.05% | 0.7473 | 30.58% | 0.0055 |
| **B3b_PreIF_DIF_Pure** | 68.16% | 60.60% | 87.59% | 71.64% | 0.7737 | 48.33% | 0.0744 |
| **B4_RandomForest_Supervised** | 91.81% | 92.02% | 89.97% | 90.98% | 0.9807 | 6.63% | 0.0034 |
| **B6a_Hybrid_Linear** | 72.13% | 70.41% | 67.79% | 69.08% | 0.7712 | 24.18% | 0.0744 |
| **B6b_Hybrid_OR** | 62.64% | 55.88% | 88.44% | 68.49% | 0.7623 | 59.27% | 0.0744 |

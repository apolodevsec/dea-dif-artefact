# Relatório de Validação Experimental e Avaliação de Baselines

**Protocolo de Validação:** 70/15/15 tráfego normal, 20/80 anomalias estratificadas (*Leakage-Free*).

## Tabela 4: Métricas Comparativas de Baselines

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 99.52% | 100.00% | 99.17% | 99.58% | 1.0000 | 0.00% | 0.0221 |
| **B2_Autoencoder_Alone** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% | 0.0000 |
| **B3_PreIF_iForest** | 97.86% | 100.00% | 96.25% | 98.09% | 1.0000 | 0.00% | 0.0241 |
| **B3b_PreIF_DIF_Pure** | 98.81% | 99.58% | 98.33% | 98.95% | 0.9997 | 0.56% | 0.6508 |
| **B4_RandomForest_Supervised** | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% | 0.0211 |
| **B6a_Hybrid_Linear** | 99.29% | 100.00% | 98.75% | 99.37% | 1.0000 | 0.00% | 0.4966 |
| **B6b_Hybrid_OR** | 99.76% | 99.59% | 100.00% | 99.79% | 0.9972 | 0.56% | 0.4966 |

## Análise de Ablação do DEAS

A cadeia comparativa de três degraus de ablação isola metodologicamente as contribuições:
1. **Espaço Latente vs. Features Brutas:** B1 (iForest Bruto) $\to$ B3 (Pre-IF Latente).
2. **Projeções Não-Lineares Neurais ($\Phi_i$):** B3 (iForest Clássico) $\to$ B3b (DIF Puro, $\lambda=0$).
3. **Impacto Específico do DEAS:** B3b (DIF $\lambda=0$) $\to$ B6a (DIF com DEAS $\lambda=0.5$ + Fusão Linear).

- **Ganho de F1-Score ($\Delta_{\text{DEAS}}^{\text{F1}}$):** `+0.0042`
- **Ganho de AUC-ROC ($\Delta_{\text{DEAS}}^{\text{AUC}}$):** `+0.0003`

## Recall Estratificado por Família de Ataque

| Família de Ataque | Total no Teste | B1_iForest_Raw (%) | B2_Autoencoder_Alone (%) | B3_PreIF_iForest (%) | B3b_PreIF_DIF_Pure (%) | B4_RandomForest_Supervised (%) | B6a_Hybrid_Linear (%) | B6b_Hybrid_OR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 144 | 98.61% | 100.00% | 93.75% | 97.22% | 100.00% | 97.92% | 100.00% |
| **Heartbleed** | 12 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **PortScan** | 84 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

## Parâmetros de Calibração sem Vazamento (*Anti-Leakage*)

| Parâmetro / Limiar | Valor Calibrado na Validação |
| :--- | :---: |
| `B1_iForest_Raw` | `0.67396` |
| `B2_Autoencoder_Alone` | `0.00609` |
| `B3_PreIF_iForest` | `0.66000` |
| `B3b_PreIF_DIF_Pure` | `0.53781` |
| `B6a_Hybrid_Linear_tau` | `0.99587` |
| `B6a_Hybrid_Linear_alpha` | `0.01000` |
| `B6b_Hybrid_OR_tau_dif` | `0.58710` |
| `B6b_Hybrid_OR_tau_ae` | `0.00609` |
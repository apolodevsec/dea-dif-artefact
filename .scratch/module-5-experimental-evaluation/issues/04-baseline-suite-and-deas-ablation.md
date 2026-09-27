# 04: Motor de Execução dos Baselines (B1 a B4) e Ablação do DEAS

**What to build:** Módulo `BaselineRunner` que orquestra o ajuste e inferência controlados dos baselines: B1 (iForest bruto), B2 (FC-DAE isolado), B3 (Pre-IF + iForest clássico), B3b (Pre-IF + DIF puro com $\lambda=0$) e B4 (Random Forest supervisionado com matriz de confusão multiclasse). Medição quantitativa da ablação $B3b (\lambda=0) \to B6a (\lambda > 0)$ isolando o ganho de detecção do DEAS.

**Blocked by:** 03-hybrid-fusion-and-adaptive-alpha

**Status:** closed

- [x] Módulo `BaselineRunner` implementado em `src/evaluation/baselines.py`.
- [x] Execução padronizada do iForest sobre features brutas (B1).
- [x] Execução padronizada do Autoencoder isolado sobre features brutas (B2).
- [x] Execução padronizada do Pre-IF com iForest clássico sobre o espaço latente (B3).
- [x] Execução do baseline de ablação B3b (DIF puro com $\lambda=0$) sobre o espaço latente.
- [x] Execução do baseline supervisionado Random Forest (B4) com métricas completas por classe.
- [x] Cálculo da métrica de ganho de ablação do DEAS $\Delta_{\text{DEAS}} = \text{F1}(B6a) - \text{F1}(B3b)$.
- [x] Teste unitário comprovando a execução de todos os baselines tabulares gerando a tabela de comparação da Tabela 4.

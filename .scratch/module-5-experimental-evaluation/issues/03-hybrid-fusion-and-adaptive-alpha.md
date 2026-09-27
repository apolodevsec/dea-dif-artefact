# 03: Fusão Híbrida e Calibração Adaptativa de Alpha (B6a e B6b)

**What to build:** Implementação da classe `HybridFusionDetector` integrando o Deep Isolation Forest com o erro MSE do FC-DAE. Inclui normalização robusta por percentis ($p_1, p_{99}$) com saturação intencional em $[0, 1]$, otimização de $\alpha \in [0, 1]$ via Youden na validação, método principal B6a (Fusão Linear Ponderada) e variante comparativa B6b (Regra Disjuntiva OR).

**Blocked by:** 02-unified-evaluator-and-stratified-recall

**Status:** closed

- [x] Classe `HybridFusionDetector` implementada em `src/evaluation/hybrid.py`.
- [x] Escalonamento robusto do erro MSE baseado nos percentis $p_1$ e $p_{99}$ da validação normal com clipping em $[0, 1]$.
- [x] Otimizador de calibração para $\alpha \in [0, 1]$ maximizando o Índice de Youden na curva ROC da validação rotulada.
- [x] Implementação de B6a: pontuação contínua $\text{Score} = \alpha \cdot s_{\text{DIF}} + (1 - \alpha) \cdot e_{\text{MSE, norm}}$.
- [x] Implementação de B6b: classificação por regra OR disjuntiva $\hat{y} = \mathbb{I}(s_{\text{DIF}} > \tau_{\text{DIF}} \lor e_{\text{MSE}} > \tau_{\text{AE}})$.
- [x] Teste de adaptividade do $\alpha$: prova de convergência $\alpha \to 1.0$ quando só DIF discrimina e $\alpha \to 0.0$ quando só MSE discrimina.
- [x] Teste de boundedness: garantia de que todos os escores gerados residem no intervalo fechado $[0, 1]$.
- [x] Teste de guarda anti-leakage: verificação de que os parâmetros de reescalonamento e $\alpha$ dependem estritamente da validação.

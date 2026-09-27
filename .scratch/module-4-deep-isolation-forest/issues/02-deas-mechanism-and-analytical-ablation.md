# 02: Mecânica DEAS e Equivalência Analítica de Ablação

**What to build:** Armazenamento de estatísticas locais nos nós das iTrees (limites $[\min, \max]$, atributo e ponto de corte $v$). Implementação da métrica DEAS com desvio contínuo relativo $dev(x, T)$ com offset $\epsilon = 10^{-7}$ e comprimento ponderado $h_{\text{DEAS}}(x)$ com fator $\lambda \in [0, 1]$. Retorno dual de escores (`score_dif_deas` e `score_dif_standard`) na interface pública `score_samples`.

**Blocked by:** 01-estimator-base-and-projection-network

**Status:** closed

- [x] Registro dos atributos de corte, valor de divisão e limites de dados em cada nó interno da árvore.
- [x] Implementação da função de desvio relativo $dev(x, T)$ com tratamento numérico para nós de variância nula ($\epsilon = 10^{-7}$).
- [x] Acúmulo do comprimento ponderado $h_{\text{DEAS}}(x) = \sum (1 - \lambda \cdot dev) + c(n)$ e normalização pelo fator de Euler $c(\psi)$.
- [x] Retorno simultâneo de dicionário contendo as chaves `score_dif_deas` e `score_dif_standard`.
- [x] Teste de invariante de ablação: comprovação estrita de que quando $\lambda = 0$, os escores DEAS e padrão são rigorosamente idênticos (`assert_allclose`).
- [x] Teste de estabilidade numérica com dados idênticos/constantes sem geração de valores `NaN` ou infinitos.
- [x] Teste de sensibilidade de $\lambda$ validando a modulação monotônica do peso do desvio local.

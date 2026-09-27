# 05: Orquestrador de Sensibilidade OFAT, CLI Unificado e Relatórios

**What to build:** Script executável `src/evaluate_protocol.py` que integra o pipeline ponta a ponta (particionamento, baselines, calibração Youden e geração de relatórios estruturados em Markdown e JSON). Módulo `OFATSensitivityAnalyzer` que executa a varredura univariada OFAT em $m, L, t, p$, registrando gráficos em `reports/figures/sensitivity/` e ressalva de não-captura de interações. Script isolado `src/baselines/lstm_ae.py` para o baseline temporal B5.

**Blocked by:** 04-baseline-suite-and-deas-ablation

**Status:** closed

- [x] Módulo `OFATSensitivityAnalyzer` em `src/evaluation/sensitivity.py` executando varreduras univariadas a partir da baseline ($m=9, L=3, t=100, p=95\%$).
- [x] Script `src/baselines/lstm_ae.py` implementando o baseline recorrente B5 sobre janelas cronológicas $W=10$.
- [x] Script CLI principal `src/evaluate_protocol.py` executando a avaliação comparativa completa dos baselines.
- [x] Geração e persistência automática de `reports/experimental_results.md` e `reports/experimental_results.json`.
- [x] Teste de integração ponta a ponta garantindo reprodutibilidade determinística de todos os resultados a partir de dataset sintético.

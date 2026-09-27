# 02: Avaliador Estatístico Unificado e Recall Estratificado por Família de Ataque

**What to build:** Implementação da classe `ExperimentalEvaluator` com cálculo das métricas padronizadas da Tabela 4 do TCC (Acurácia, Precisão, Recall, F1-Score, AUC-ROC, FPR e latência média por fluxo) e computação da tabela de Taxa de Detecção (Recall) estratificada por classe/família de ataque ($\text{Recall}_k$).

**Blocked by:** 01-leakage-free-stratified-partitioner

**Status:** closed

- [x] Classe `ExperimentalEvaluator` implementada em `src/evaluation/evaluator.py`.
- [x] Cálculo das métricas binárias padronizadas: Acurácia, Precisão, Recall, F1-Score, AUC-ROC e Taxa de Falsos Positivos (FPR).
- [x] Cálculo do Recall estratificado por classe de ataque individual ($\text{Recall}_k$) para detectores binários.
- [x] Registro e formatação de telemetria de latência média de inferência por amostra (ms/fluxo).
- [x] Teste unitário de corretude estatística com valores de matriz de confusão conhecidos.
- [x] Teste unitário de que a taxa de detecção estratificada bate com a contagem manual de verdadeiros positivos por classe.

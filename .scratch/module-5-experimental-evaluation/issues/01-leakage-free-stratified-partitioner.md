# 01: Particionador Estratificado Livre de Vazamento e Guarda Anti-Leakage

**What to build:** Implementação da classe `DataPartitioner` responsável por particionar tráfego legítimo normal em 70% treino, 15% validação e 15% teste cego, e amostras anômalas em 20% validação (para calibração supervisionada de Youden) e 80% teste cego com estratificação por classe de ataque. Identifica classes raras (< 15 instâncias) com ressalva estatística e garante determinismo bit a bit via semente aleatória.

**Blocked by:** None (can start immediately)

**Status:** closed

- [x] Classe `DataPartitioner` implementada em `src/evaluation/partitioner.py`.
- [x] Particionamento do tráfego normal em 70/15/15 e anomalias em 20/80 com estratificação por rótulo.
- [x] Invariante de integridade: a partição de treino normal não contém nenhuma amostra anômala (0% de ataques).
- [x] Invariante de disjunção mútua: conjuntos de treino, validação e teste possuem interseção estritamente vazia de índices.
- [x] Identificação automática de classes raras (< 15 amostras) gerando dicionário de alertas de significância amostral.
- [x] Teste de determinismo comprovando que sementes idênticas produzem partições rigorosamente idênticas.

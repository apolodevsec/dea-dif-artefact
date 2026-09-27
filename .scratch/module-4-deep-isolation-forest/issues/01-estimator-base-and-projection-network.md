# 01: Estimador Base DIF e Rede de Projeção Aleatória

**What to build:** Implementação inicial da classe `DeepIsolationForest` com interface pública `fit(X)` e `score_samples(X)`, acoplada a uma rede feedforward de pesos congelados em PyTorch com inicialização Kaiming Normal e ativação `LeakyReLU(0.2)`. A árvore de isolamento particiona subamostras de 256 instâncias normais e computa o comprimento de caminho clássico $h(x)$.

**Blocked by:** None (can start immediately)

**Status:** closed

- [x] Classe `DeepIsolationForest` implementada em `src/models/dif.py` com interface `fit(X)` e `score_samples(X)`.
- [x] Rede de projeção neural feedforward implementada com camadas densas, pesos congelados (`requires_grad=False`), ativação `LeakyReLU(0.2)` e inicialização Kaiming Normal.
- [x] Construção de árvores de isolamento recursivas com profundidade máxima $h_{\max} = 8$ sobre subamostras $\psi = 256$.
- [x] Cálculo do comprimento de caminho médio e escore de isolamento clássico normalizado em $[0, 1]$.
- [x] Teste unitário comprovando que o ajuste opera unicamente sobre dados legítimos (sem exigir parâmetro de contaminação prévia).
- [x] Teste unitário de integridade dimensional validando entrada $z \in \mathbb{R}^9$ e conformidade do escore emitido.

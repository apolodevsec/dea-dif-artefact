# 04: Desacoplamento de Representações e Vetorização de Alta Performance

**What to build:** Gestão do pool de $r$ redes de projeção atendendo $t/r$ árvores cada ($r=10$ padrão e $r=t=100$ no modo estrito). Projeção de representações em mini-batches via PyTorch, percurso vetorizado por máscaras booleanas por nível de profundidade em NumPy e paralelização multi-core de árvores com `joblib(n_jobs=-1)`.

**Blocked by:** 02-deas-mechanism-and-analytical-ablation

**Status:** closed

- [x] Suporte à parametrização de $r$ redes neurais de representação alimentando $t/r$ árvores com subamostras independentes.
- [x] Projeção de tensores latentes em mini-batches na GPU/CPU via PyTorch (`batch_size=32768`).
- [x] Percurso de árvore vetorizado em arrays NumPy através de máscaras booleanas por nível de profundidade recursivo/iterativo.
- [x] Paralelismo multi-core de inferência nas árvores através de `joblib.Parallel`.
- [x] Teste de invariância de lote: comprovação de que inferência dividida em batches produz escores idênticos ao lote único.
- [x] Teste de determinismo e reprodutibilidade de inferência controlado por semente aleatória (`random_state`).
- [x] Teste de benchmark de throughput atestando alta vazão em grandes lotes de dados.

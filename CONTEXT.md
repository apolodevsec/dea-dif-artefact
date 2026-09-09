# Glossário do Domínio

## Módulo 2 — FC-DAE (Fully Connected Deep Autoencoder)

### FC-DAE (Fully Connected Deep Autoencoder)
Rede neural profunda totalmente conectada com arquitetura simétrica (Encoder-Bottleneck-Decoder). É responsável por aprender a distribuição latente do tráfego normal de rede em um regime de aprendizado não-supervisionado.

### Espaço Latente ($z$)
Representação vetorial comprimida de dimensão $m$ gerada na camada *bottleneck* do Encoder com ativação linear. Preserva a continuidade geométrica dos dados e serve como entrada direta para o módulo subsequente (Deep Isolation Forest).

### Limiar de Anomalia ($\tau_{AE}$)
Valor de corte estatístico aplicado ao Erro Quadrático Médio (MSE) de reconstrução. É determinado com base no tráfego normal de validação ($\mu_{\text{val}} + 3\sigma_{\text{val}}$ ou percentil 99), definindo a fronteira de rejeição para desvios de reconstrução.

### Vetor de Entrada ($x$)
Vetor de $n$ características contínuas extraídas de conexões de rede, pré-processadas e normalizadas no intervalo $[0, 1]$ (armazenadas em formato `.parquet` derivadas do dataset CICIDS2017 ou NSL-KDD).

### Espaço Latente Persistido
Conjunto de dados exportado em formato `.parquet` contendo os vetores de espaço latente $z \in \mathbb{R}^m$ acompanhados dos rótulos de classe originais. Destina-se ao consumo do Deep Isolation Forest (Módulo 4) e análises de separabilidade geométrica via t-SNE.

---

## Módulo 3 — Integração Pre-IF (Pipeline de Transição)

### Pre-IF (Pre-Isolation Forest)
Módulo de transição que isola a camada *bottleneck* do FC-DAE como extrator independente, realizando a validação de separabilidade geométrica e preparando os vetores latentes para ingestão pelo Deep Isolation Forest.

### Análise de Separabilidade via t-SNE
Técnica de projeção não-linear utilizada para mapear o espaço latente de $m$ dimensões para um plano 2D, avaliando visualmente a formação de *clusters* distintos entre o tráfego normal e anômalo.

### Análise de Reconstrução Diferencial
Avaliação comparativa da distribuição empírica dos erros de reconstrução MSE entre o tráfego benigno e os diferentes tipos de ataques, comprovando a sensibilidade do modelo a desvios de padrão.

# 1. Arquitetura FC-DAE com Bottleneck Linear para Extração do Espaço Latente em PyTorch

Data: 2026-09-09

## Status

Aceito

## Contexto

O Módulo 2 do framework híbrido necessita comprimir vetores de características de tráfego de rede $x \in \mathbb{R}^n$ em representações latentes $z \in \mathbb{R}^m$, preservando a topologia e a continuidade geométrica dos dados normais para posterior processamento pelo Deep Isolation Forest (Módulo 4).

## Decisão

Adotar a arquitetura **Fully Connected Deep Autoencoder (FC-DAE)** desenvolvida em **PyTorch** com as seguintes diretrizes:
1. Topologia estritamente simétrica de 3 camadas no Encoder ($n \rightarrow n/2 \rightarrow n/4 \rightarrow m$) e 3 no Decoder ($m \rightarrow n/4 \rightarrow n/2 \rightarrow n$).
2. Ativação ReLU nas camadas ocultas intermediárias para mitigar desaparecimento de gradiente e ativação **Linear** na camada *bottleneck* para garantir a continuidade do espaço latente.
3. Dimensionamento dinâmico do *bottleneck*: $m = \lfloor n/4 \rfloor$ para $n < 50$ e $m = \lfloor n/8 \rfloor$ para $n \ge 50$.
4. Organização do código com métodos explícitos `encode(x)`, `decode(z)` e `forward(x)`.
5. Treinamento estritamente não-supervisionado em PyTorch (otimizador Adam, `ReduceLROnPlateau`, Parada Antecipada com paciência 10).
6. Persistência dos vetores de espaço latente $z$ extraídos em formato `.parquet` acompanhados dos rótulos originais para viabilizar inspeção geométrica via t-SNE.

## Consequências

### Positivas
- A transparência da API dinâmica do PyTorch permite a extração limpa e direta de $z$ via `encode(x)`.
- A persistência em arquivos `.parquet` garante desacoplamento completo entre os módulos do pipeline.
- A preservação da estrutura não-linear no *bottleneck* sem saturação facilita a separação geométrica posterior pelo Deep Isolation Forest.

### Negativas
- Exige atenção na padronização da entrada em $[0, 1]$ devido à ativação Sigmoide na camada final de saída do Decoder.

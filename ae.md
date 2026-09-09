# Detalhamento Técnico: Módulo 2 — Arquitetura do Deep Autoencoder

Este documento detalha tecnicamente a **Seção 4.4** do documento base, que descreve a arquitetura e o funcionamento do Módulo 2 do framework híbrido proposto para detecção de intrusões [cite: 1]. 

O componente projetado é uma rede neural profunda totalmente conectada, denominada **Fully Connected Deep Autoencoder (FC-DAE)** [cite: 1]. O modelo adota uma topologia estritamente simétrica composta por *encoder-bottleneck-decoder*, garantindo que a capacidade expressiva do *Decoder* para a reconstrução dos dados seja equivalente à capacidade de compressão do *Encoder* [cite: 1].

## 4.4.1 Arquitetura do Encoder

A função do *Encoder* é mapear o vetor de características de entrada $x \in \mathbb{R}^n$ para uma representação latente comprimida $z \in \mathbb{R}^m$, onde a dimensionalidade é significativamente reduzida ($m \ll n$) [cite: 1]. A arquitetura é construída com três camadas densas que aplicam uma redução progressiva de dimensionalidade [cite: 1]:

*   **Camada de Entrada:** Recebe as $n$ *features* do tráfego [cite: 1].
*   **Camada Oculta 1:** Contém $n/2$ neurônios e utiliza a função de ativação ReLU [cite: 1].
*   **Camada Oculta 2:** Contém $n/4$ neurônios e utiliza a função de ativação ReLU [cite: 1].
*   **Camada *Bottleneck* (Espaço Latente):** Contém $m$ neurônios (onde $m$ tipicamente varia no intervalo $[n/8, n/4]$) e utiliza uma **ativação linear** [cite: 1].

**Justificativas de Projeto:**
*   **Ativação ReLU:** Adotada nas camadas intermediárias devido às suas propriedades de esparsidade e para contornar o problema de desaparecimento do gradiente (*vanishing gradient*) [cite: 1].
*   **Ativação Linear no Bottleneck:** A ausência de ativação não-linear na camada latente tem o propósito de preservar a continuidade do espaço latente [cite: 1]. Isso facilita a separação geométrica entre as representações das amostras normais e anômalas, otimizando o trabalho do módulo de isolamento subsequente [cite: 1].

## 4.4.2 Arquitetura do Decoder e Função de Custo

O *Decoder* executa a transformação matemática inversa, visando reconstruir o vetor original a partir da representação latente $z$ [cite: 1]. 

*   **Topologia:** Segue o caminho inverso estritamente simétrico, escalonando de $m \rightarrow n/4 \rightarrow n/2 \rightarrow n$ neurônios [cite: 1].
*   **Camada de Saída:** Aplica a função de ativação **sigmoide**, adequada para dados que foram previamente submetidos à normalização Min-Max [0, 1] no Módulo 1 [cite: 1].
*   **Função de Custo:** O treinamento ocorre pela minimização do Erro Quadrático Médio (MSE), expresso pela fórmula:
    $$\mathcal{L}(x,\hat{x}) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{x}_i)^2$$ [cite: 1]

**Implementação e Otimização:**
*   A implementação será realizada utilizando o *framework* **PyTorch** (aproveitando grafos dinâmicos e aceleração GPU) [cite: 1].
*   O treinamento utiliza o otimizador **Adam** com uma taxa de aprendizado inicial de $10^{-3}$ [cite: 1].
*   Para evitar estagnação, implementa-se um escalonador de taxa de aprendizado (*scheduler* `ReduceLROnPlateau`) que reduz a taxa por um fator definido quando a perda no conjunto de validação para de cair por um número fixo de épocas [cite: 1].

## 4.4.3 Protocolo de Treinamento e Estratégia de Detecção

O protocolo de treinamento do FC-DAE baseia-se num regime estritamente não-supervisionado [cite: 1].

*   **Dataset Principal:** Arquivos em formato `.parquet` já pré-processados contendo o dataset **CICIDS2017**, mantendo o **NSL-KDD** como benchmark secundário.
*   **Treinamento:** O modelo é treinado **exclusivamente com amostras de tráfego normal** [cite: 1]. Adota-se uma divisão de 80% dos dados para treinamento e 20% para validação, com o uso da técnica de parada antecipada (*early stopping*) configurada com uma paciência de 10 épocas para mitigar *overfitting* [cite: 1].
*   **Dimensionamento de Bottleneck:** Definido dinamicamente como $m = \lfloor n/4 \rfloor$ para $n < 50$, e $m = \lfloor n/8 \rfloor$ para $n \ge 50$.
*   **Inferência e Extração Latente:** O Erro Quadrático Médio (MSE) é monitorado com o limiar de anomalia calibrado por $\tau_{AE} = \mu_{\text{val\_MSE}} + 3\sigma_{\text{val\_MSE}}$ (ou percentil 99 do conjunto de validação normal) [cite: 1]. Nesta etapa inicial, o foco principal da arquitetura é a **extração direta da representação comprimida do espaço latente ($z$)** a partir do *bottleneck*, repassando este vetor para processamento no módulo posterior (Deep Isolation Forest) [cite: 1].
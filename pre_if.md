# Detalhamento Técnico: Módulo 3 — Integração Pre-IF: Pipeline de Transição

Este documento detalha tecnicamente a **Seção 4.5** do documento base, que descreve a arquitetura e o funcionamento do Módulo 3 do framework híbrido proposto para detecção de intrusões [cite: 1].

O módulo de integração **Pre-IF** constitui a interface de transição entre o Autoencoder (Módulo 2) e o Deep Isolation Forest (Módulo 4). Sua função é estruturar o mapeamento e a preparação dos dados do espaço latente $z \in \mathbb{R}^m$, garantindo que as propriedades topológicas e geométricas sejam preservadas para o estágio de isolamento [cite: 1].

## 4.5.1 Extração e Validação do Espaço Latente

Após a conclusão do treinamento não-supervisionado do FC-DAE, a camada *bottleneck* é isolada como um extrator de características independente [cite: 1]. Para cada vetor de entrada $x \in \mathbb{R}^n$, o Encoder computa a representação reduzida $z = \text{Encoder}(x) \in \mathbb{R}^m$ [cite: 1]. 

A qualidade da representação gerada no espaço latente é validada empiricamente por dois procedimentos complementares [cite: 1]:

1.  **Análise de Separabilidade via t-SNE:** Aplicação do algoritmo de alocação estocástica de vizinhos com distribuição t (*t-Distributed Stochastic Neighbor Embedding*) sobre uma amostra representativa do espaço latente [cite: 1]. O objetivo é verificar visualmente se as representações de amostras normais e anômalas formam agrupamentos (*clusters*) geometricamente bem separados [cite: 1].
2.  **Análise de Reconstrução Diferencial:** Comparação das distribuições dos erros de reconstrução (MSE) gerados pelo Autoencoder para amostras normais versus amostras anômalas [cite: 1]. Espera-se uma clara divergência estatística, onde amostras anômalas apresentam erros significativamente superiores devido à ausência desses padrões no conjunto de treinamento não-supervisionado [cite: 1].

## 4.5.2 Vantagens da Abordagem Pre-IF sobre Features Brutas

A substituição do espaço original de características $x \in \mathbb{R}^n$ pelo espaço latente $z \in \mathbb{R}^m$ proporciona três vantagens estruturais cruciais para o framework híbrido [cite: 1]:

*   **Redução de Dimensionalidade:** Diminui a complexidade computacional e o consumo de memória na fase de construção e inferência do Deep Isolation Forest [cite: 1].
*   **Redução de Ruído:** Atua como um filtro denso, eliminando variáveis ruidosas, irrelevantes ou altamente correlacionadas presentes no tráfego bruto [cite: 1].
*   **Pré-separação Geométrica:** A compressão pelo Encoder com ativação linear no *bottleneck* contorce o espaço de dados de forma a tornar os limites entre o tráfego normal e as anomalias mais nítidos, otimizando a eficiência das partições aleatórias executadas pelo DIF [cite: 1].

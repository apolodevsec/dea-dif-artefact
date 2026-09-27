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

---

## Módulo 4 — Componente Deep Isolation Forest (DIF)

### Deep Isolation Forest (DIF)
Algoritmo de detecção de anomalias baseado em isolamento que projeta o espaço latente $z$ através de um ensemble de redes neurais com pesos aleatórios, aplicando particionamento recursivo por eixos paralelos nas representações resultantes.

### Rede de Projeção Aleatória ($\Phi_i$)
Rede neural feedforward de $L$ camadas com pesos inicializados aleatoriamente e não treináveis que projeta o espaço latente $z \in \mathbb{R}^m$ em uma representação não-linear intermediária $z_i \in \mathbb{R}^d$, viabilizando superfícies de isolamento não-lineares arbitrárias no espaço original.

### Score DEAS (Deviation-Enhanced Anomaly Scoring)
Função de pontuação de isolamento que combina a profundidade percorrida na árvore com o desvio quantitativo contínuo dos dados em relação aos limiares de corte dos nós, enriquecendo o comprimento do caminho com informação da densidade local.

### Dimensão de Projeção ($d$)
Dimensionalidade do espaço intermediário $z_i$ gerado pela rede $\Phi_i$ onde os cortes axiais do iTree são realizados, ajustável entre $m$ e $\lfloor m/2 \rfloor$.

### Conjunto de Representações ($r$)
Número de redes neurais de projeção aleatória independentes instanciadas no ensemble DIF ($r \le t$), onde cada rede projeta subamostras para um conjunto de $t/r$ árvores de isolamento.
_Evitar_: Número de modelos, redes treinadas.

### Regime de Treinamento Exclusivamente Benigno
Diretriz de calibração do DIF em que as árvores de isolamento são induzidas estritamente sobre vetores latentes de conexões legítimas/normais, eliminando a suposição de contaminação prévia no treino.
_Evitar_: Treinamento supervisionado, contaminação arbitrária.

### Score Tradicional de Isolamento
Pontuação baseada unicamente no comprimento médio discreto de caminho $h(x)$ do Isolation Forest clássico, calculada e persistida em paralelo para viabilizar o estudo de ablação do DEAS.
_Evitar_: Score sem rede, score não-profundo.

### Fator de Ponderação DEAS ($\lambda$)
Coeficiente de sensibilidade $\lambda \in [0, 1]$ que modula o impacto do desvio contínuo local no comprimento da aresta de isolamento. Quando $\lambda = 0$, o modelo colapsa exatamente no cálculo de isolamento clássico.
_Evitar_: Taxa de aprendizado, penalidade L2.

### Limiar de Referência do DIF ($\tau_{\text{DIF, ref}}$)
Valor de corte estatístico não-supervisionado calculado a partir de um percentil pré-definido sobre os scores de anomalia do tráfego normal de validação no Módulo 4.
_Evitar_: Limiar ótimo, ponto de Youden.

### Ponto de Operação Supervisionado ($\tau_{\text{DIF, opt}}$)
Limiar de classificação binária otimizado no Módulo 5 através da curva ROC e do Índice de Youden utilizando o conjunto de validação com tráfego benigno e anômalo rotulado.
_Evitar_: Limiar de treino, percentil não-supervisionado.

---

## Módulo 5 — Avaliação Experimental e Baselines

### Score Híbrido de Detecção
Pontuação contínua resultante da combinação linear ponderada do score do Deep Isolation Forest com o erro de reconstrução normalizado do Autoencoder: $\text{Score}_{\text{final}} = \alpha \cdot s_{\text{DIF}} + (1 - \alpha) \cdot e_{\text{MSE, norm}}$, onde $\alpha \in [0, 1]$ é calibrado na validação.
_Evitar_: Média simples, soma não-normalizada.

### Regra de Decisão Disjuntiva (Regra OR)
Critério de detecção híbrido em que uma conexão é classificada como ataque se $s_{\text{DIF}} > \tau_{\text{DIF}}$ OU $e_{\text{MSE}} > \tau_{\text{AE}}$, permitindo que cada detector atue como mecanismo de veto especializado.
_Evitar_: Classificação única, threshold adaptativo fixo.

### Taxa de Detecção Estratificada
Métrica que computa o Recall individual para cada família ou categoria específica de ataque presente no conjunto de teste, avaliando a cobertura do detector binário não-supervisionado sem assumir classificação multinomial.
_Evitar_: F1 multiclasse não-supervisionado, matriz de confusão multiclasse NxN.

### Baseline de Ablação DIF Puro (Baseline 3b)
Modelo de controle composto pelo FC-DAE + DIF operando com fator de sensibilidade $\lambda = 0$, que desativa o DEAS e preserva apenas o efeito das projeções neurais aleatórias $\Phi_i$, permitindo isolar empiricamente a contribuição científica de cada componente.
_Evitar_: Modelo sem autoencoder, iForest simples.

### Particionamento Estratificado com Ataques na Validação
Divisão metodológica que reserva uma fração estratificada dos ataques no conjunto de validação (ex.: 20% val / 80% teste) para calibrar o ponto de operação supervisionado de Youden ($\tau_{\text{opt}}$) e o peso $\alpha$ sem incorrer em vazamento de dados do conjunto de teste cego.
_Evitar_: 100% de ataques no teste, validação sem anomalias.

### Normalização Robusta do Erro MSE
Escalonamento do erro de reconstrução baseado em percentis de corte robustos ($p_1$ e $p_{99}$ da validação) com saturação intencional (*clipping*) no intervalo $[0, 1]$, evitando que outliers inflem a escala e desequilibrem a fusão linear.
_Evitar_: Min-Max absoluto sem clipping, escala livre.

### Janela Temporal com Lookback
Estratégia de modelagem sequencial para o baseline LSTM-AE em que uma janela de $W$ fluxos consecutivos atribui a predição ao fluxo final ($t_W$), conferindo contexto histórico assimétrico em relação aos detectores puramente pontuais.
_Evitar_: Janela causal centrada, classificação de pacote único.

### Análise de Sensibilidade OFAT
Metodologia univariada (*One-Factor-at-a-Time*) que varia um hiperparâmetro por vez a partir da baseline fixa ($m=9, L=3, t=100, p=95\%$), reduzindo o espaço de busca experimental com a limitação explícita de não mapear interações cruzadas de segunda ordem.
_Evitar_: Grid search fatorial completo, busca aleatória.






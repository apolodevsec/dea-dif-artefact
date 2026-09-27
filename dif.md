# Detalhamento Técnico: Módulo 4 — Componente Deep Isolation Forest (DIF)

Este documento detalha tecnicamente a **Seção 4.6** do documento base do framework híbrido proposto para detecção de intrusões (NIDS).

O componente **Deep Isolation Forest (DIF)** opera diretamente sobre os vetores de espaço latente $z \in \mathbb{R}^m$ (onde $m=9$), persistidos pelo pipeline de transição Pre-IF (Módulo 3). Sua função é gerar os escores de anomalia baseados no isolamento espacial em representações não-lineares projetadas aleatoriamente, mitigando o problema de *ghost regions* do Isolation Forest tradicional através do enriquecimento DEAS (*Deviation-Enhanced Anomaly Scoring*).

---

## 4.6.1 Arquitetura do Ensemble e Redes de Projeção ($\Phi_i$)

O ensemble DIF constrói $t = 100$ árvores de isolamento a partir de um conjunto de redes neurais feedforward de pesos aleatórios e congelados.

*   **Conjunto de Representações ($r$):** Instancia-se um conjunto parametrizável de $r \le t$ redes neurais de projeção independentes ($\Phi_1, \dots, \Phi_r$). Por padrão de engenharia, adota-se $r = 10$ redes, onde cada representação atende a $t/r = 10$ árvores com subamostragens independentes ($\psi = 256$). O modo estrito ($r = t = 100$) é parametrizável via CLI (`--num_representations 100`).
*   **Topologia de $\Phi_i$:** Cada rede possui $L = 3$ camadas densas (com suporte experimental a $L \in [2, 4]$) com a seguinte configuração:
    $$\mathbb{R}^m \xrightarrow{\text{Linear}} \mathbb{R}^{2m} \xrightarrow{\text{LeakyReLU}} \mathbb{R}^{2m} \xrightarrow{\text{LeakyReLU}} \mathbb{R}^d$$
*   **Função de Ativação:** `LeakyReLU(negative_slope=0.2)` nas camadas ocultas, introduzindo não-linearidade sem saturação dos gradientes nem colapso a zero de valores negativos.
*   **Dimensionalidade de Projeção ($d$):** Dimensão de saída $d = 9 = m$ (com suporte experimental a $d = 4 \approx \lfloor m/2 \rfloor$).
*   **Inicialização dos Pesos:** Inicialização ortogonal/Kaiming Normal com média 0 e desvio padrão fixado. Os pesos são estritamente congelados (`requires_grad = False`), garantindo regime sem treinamento ou retropropagação.

---

## 4.6.2 Indução das Árvores de Isolamento (iTrees)

Para cada uma das $t = 100$ árvores do ensemble:
1.  **Subamostragem:** Amostram-se aleatoriamente $\psi = 256$ vetores latentes exclusivamente a partir do conjunto de tráfego legítimo (normal).
2.  **Mapeamento Não-Linear:** O lote de amostras é projetado no espaço intermediário $z_i = \Phi_i(z) \in \mathbb{R}^d$.
3.  **Particionamento Recursivo:**
    *   Sorteia-se aleatoriamente um eixo $j \in \{1, \dots, d\}$.
    *   Determinam-se os limites extremos $[\min(z_{i, j}), \max(z_{i, j})]$ das amostras que alcançaram o nó.
    *   Sorteia-se um ponto de corte uniforme $v \sim \mathcal{U}(\min(z_{i, j}), \max(z_{i, j}))$.
    *   Os dados são divididos em subconjuntos esquerdo ($z_{i, j} < v$) e direito ($z_{i, j} \ge v$).
4.  **Critérios de Parada:** A recursão é encerrada quando a profundidade atinge o limite máximo $h_{\max} = \lceil \log_2(\psi) \rceil = 8$, quando $|X_{\text{nó}}| \le 1$ ou quando todas as amostras no nó forem idênticas.

---

## 4.6.3 Cálculo de Anomalia e Formulação do DEAS

> [!NOTE]
> **Transparência Científica / Alinhamento de Referência:**
> A formulação abaixo é uma **variante operacional inspirada na mecânica do DEAS de Xu et al. (2023)**. Ela preserva a propriedade analítica fundamental de redutibilidade: quando o fator $\lambda = 0$, a fórmula colapsa exatamente no algoritmo clássico do Isolation Forest (Liu et al., 2012).

Em cada nó interno $T$ percorrido por uma amostra $x$, computa-se o desvio relativo contínuo do ponto projetado em relação ao limiar de corte:
$$dev(x, T) = \frac{|z_{i, j} - v|}{\max(z_{i, j}) - \min(z_{i, j}) + \epsilon}$$
onde $\epsilon = 10^{-7}$ previne divisão por zero.

### Comprimento de Caminho Ponderado ($h_{\text{DEAS}}$)
O comprimento de caminho efetivo percorrido até o nó folha é calculado por:
$$h_{\text{DEAS}}(x) = \sum_{k=1}^{\text{prof}} \left(1 - \lambda \cdot dev_k(x)\right) + c(n_{\text{folha}})$$
onde:
*   $\lambda \in [0, 1]$ é o fator de ponderação de desvio (padrão $\lambda = 0.5$).
*   $n_{\text{folha}}$ é o número de amostras remanescentes na folha atingida.
*   $c(n)$ é o comprimento médio de busca mal-sucedida em BST para $n$ amostras:
    $$c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n} \quad (\text{para } n > 2)$$
    com $c(2) = 1$ e $c(n \le 1) = 0$.

### Score de Anomalia Normalizado
O escore de anomalia da amostra no intervalo $[0, 1]$ é computado pela média sobre as $t$ árvores:
$$s_{\text{DEAS}}(x) = 2^{-\frac{\mathbb{E}[h_{\text{DEAS}}(x)]}{c(\psi)}}$$

---

## 4.6.4 Tabela 3 Atualizada — Hiperparâmetros do Componente DIF

A tabela a seguir expande a Tabela 3 original do TCC, incorporando as definições de implementação necessárias para reprodutibilidade estrita:

| Hiperparâmetro | Valor Proposto / Padrão | Intervalo / Opções | Justificativa / Referência |
| :--- | :---: | :---: | :--- |
| **Número de árvores ($t$)** | `100` | $t \ge 100$ | Estabilidade estatística do escore (Liu et al., 2012) |
| **Tamanho da subamostra ($\psi$)** | `256` | $256$ | Padrão ótimo contra *swamping* e *masking* (Liu et al., 2012) |
| **Conjunto de representações ($r$)** | `10` | $[1, t]$ ($10$ ou $100$) | Redução de complexidade de inferência e footprint de memória |
| **Camadas da rede ($\Phi_i$) ($L$)** | `3` | $[2, 4]$ | Não-linearidade suficiente sem dispersão descontrolada |
| **Dimensão de projeção ($d$)** | `9` ($=m$) | $\{4, 9\}$ | Preservação da capacidade informacional do *bottleneck* linear |
| **Função de ativação** | `LeakyReLU(0.2)` | `LeakyReLU`, `Tanh` | Não-linearidade sem saturação nem corte abrupto de negativos |
| **Inicialização de pesos** | `Kaiming Normal` | Pesos fixos | Garantia de variância constante das ativações sem treino |
| **Treinamento de $\Phi_i$** | `Não` (`requires_grad=False`) | Congelados | Princípio central de projeções aleatórias no DIF |
| **Fator de modulação DEAS ($\lambda$)** | `0.5` | $[0, 1]$ | Permite ablação analítica estrita com iForest tradicional ($\lambda=0$) |
| **Profundidade máxima ($h_{\max}$)** | `8` | $\lceil \log_2(\psi) \rceil$ | Limite teórico de expansão para subamostra de 256 |

---

## 4.6.5 Protocolo de Calibração e Persistência de Saídas

1.  **Regime Exclusivamente Benigno:** O ensemble é treinado em `src/train_dif.py` ingerindo apenas os registros onde `label == 0` (ou tráfego normal) da partição de treino.
2.  **Limiar de Referência Não-Supervisionado ($\tau_{\text{DIF, ref}}$):**
    *   Calculado e persistido nos metadados do modelo (`models/dif_metadata.json` e `models/dif_ensemble.joblib`).
    *   Definido estatisticamente sobre o conjunto de validação normal pelo percentil $p$ fixado (alinhado com o protocolo de $p=95$ ou $p=99$).
3.  **Ponto de Operação Supervisionado ($\tau_{\text{DIF, opt}}$):**
    *   Reservado para cálculo no **Módulo 5**, onde curvas ROC, Precision-Recall e o critério de Youden serão aplicados sobre validação mista (rotulada).
4.  **Artefato Persistido de Inferência (`data/processed/dif_scores.parquet`):**
    O script `src/score_dif.py` processa as 1.589.924 amostras e persiste um arquivo contendo:
    *   `index`: Índice original do fluxo de rede.
    *   `score_dif_deas`: Pontuação de isolamento com ponderação DEAS ($\lambda = 0.5$).
    *   `score_dif_standard`: Pontuação de isolamento tradicional sem ponderação ($\lambda = 0$).
    *   `label_binary`: Rótulo verdadeiro da conexão ($0 = \text{Benigno}, 1 = \text{Ataque}$).
    *   `label_multiclass`: Categoria específica da conexão para avaliação detalhada.

---

## 4.6.6 Checklist de Alinhamento com o Texto Acadêmico (TCC)

Para assegurar 100% de coerência entre a implementação e o texto do trabalho acadêmico:
- [ ] **Seção 2.4.1:** Harmonizar a menção de "$t$ redes neurais distintas" com a parametrização de "$r$ representações aleatórias compartilhadas entre subconjuntos de árvores".
- [ ] **Seção 2.4.2:** Registrar explicitamente a fórmula adotada do DEAS como uma variante de ponderação analítica inspirada em Xu et al. (2023), destacando o parâmetro $\lambda$.
- [ ] **Tabela 3:** Adicionar as linhas de *Conjunto de Representações ($r$)*, *Função de Ativação*, *Inicialização de Pesos* e *Fator $\lambda$ do DEAS*.
- [ ] **Seção 4.7.1 e 4.7.2:** Formalizar a distinção entre $\tau_{\text{DIF, ref}}$ (percentil não-supervisionado no Módulo 4) e $\tau_{\text{DIF, opt}}$ (índice de Youden no Módulo 5), alinhando o percentil base utilizado.

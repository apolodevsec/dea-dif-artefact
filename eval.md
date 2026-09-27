# Detalhamento Técnico: Módulo 5 — Protocolo de Validação Experimental e Avaliação de Baselines

Este documento detalha tecnicamente a **Seção 4.8** do documento base do framework híbrido proposto para detecção de intrusões em redes (NIDS).

O objetivo do Módulo 5 é orquestrar a avaliação empírica rigorosa, executando comparações controladas contra baselines canônicos da literatura, conduzindo a análise de ablação do componente DEAS e gerando as métricas padronizadas de detecção sobre o conjunto de teste cego.

---

## 4.8.1 Datasets e Protocolo de Particionamento sem Vazamento

O framework híbrido foi concebido para validação empírica em três bases de referência: **CICIDS2017**, **NSL-KDD** e **UNSW-NB15**. 

### Esquema de Particionamento Estatístico
Para evitar contaminação (*data leakage*) e possibilitar a calibração legítima do limiar operacional ótimo via Índice de Youden, o particionamento divide amostras legítimas e anômalas da seguinte forma:

| Partição | Amostras Normais | Amostras de Ataque | Finalidade Metodológica |
| :--- | :---: | :---: | :--- |
| **Treinamento** | `70%` | `0%` (0 instâncias) | Ajuste estritamente não-supervisionado do FC-DAE e do DIF. |
| **Validação** | `15%` | `20%` (estratificado) | Calibração de $\tau_{\text{ref}}$, otimização de $\alpha$ e calibração de Youden ($\tau_{\text{opt}}$). |
| **Teste Cego** | `15%` | `80%` (estratificado) | Avaliação final cega e extração das métricas oficiais de desempenho. |

> [!NOTE]
> **Ressalva de Significância Amostral para Classes Ultra-Raras:**  
> Classes de ataque com baixíssima frequência (ex.: *Heartbleed* com 11 instâncias no CICIDS2017 e *U2R* com ~52 instâncias no NSL-KDD) têm seu split estratificado mantido (2 a 7 instâncias na validação), mas o Recall reportado na avaliação final é acompanhado de ressalva estatística explícita quanto à amplitude do intervalo de confiança amostral.

---

## 4.8.2 Cadeia Comparativa de Baselines e Degraus de Ablação

A tabela comparativa é estruturada em três degraus progressivos para isolar a contribuição científica de cada componente:

| ID | Modelo / Baseline | Espaço de Entrada | DEAS ($\lambda$) | Finalidade Científica |
| :---: | :--- | :---: | :---: | :--- |
| **B1** | iForest Isolado | Features Brutas ($x \in \mathbb{R}^n$) | `Não` | Baseline tradicional da literatura sobre dados originais. |
| **B2** | FC-DAE Isolado | Features Brutas ($x \in \mathbb{R}^n$) | `N/A` | Avaliação da capacidade de detecção por erro MSE isolado. |
| **B3** | Pre-IF + iForest Clássico | Espaço Latente ($z \in \mathbb{R}^m$) | `Não` | Efeito da compressão latente sobre o iForest clássico. |
| **B3b** | **Pre-IF + DIF Puro (Ablação)** | Espaço Latente ($z \in \mathbb{R}^m$) | `Não` ($\lambda=0$) | **Isola o ganho das projeções neurais $\Phi_i$ sobre o iForest.** |
| **B4** | Random Forest Supervisionado | Features Brutas ($x \in \mathbb{R}^n$) | `N/A` | Teto de referência supervisionado clássico (com rótulos no treino). |
| **B5** | LSTM Autoencoder (LSTM-AE) | Janelas Temporais de Fluxos | `N/A` | Referência de modelagem de dependências sequenciais recorrentes. |
| **B6a** | Pre-IF + DIF com DEAS | Espaço Latente ($z \in \mathbb{R}^m$) | `Sim` ($\lambda=0.5$) | **Isola o ganho incremental do DEAS na mitigação de ghost regions.** |
| **B6b** | **Framework Híbrido Completo** | Latente + Erro MSE | `Sim` | **Modelo proposto final com Fusão Ponderada / Regra OR.** |

---

## 4.8.3 Métricas de Avaliação e Classificação por Categoria

A avaliação consolida as métricas padronizadas de NIDS calculadas no conjunto de teste cego:

$$\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN} \qquad \text{Precisão} = \frac{VP}{VP + FP}$$
$$\text{Recall} = \frac{VP}{VP + FN} \qquad \text{F1-Score} = \frac{2 \cdot P \cdot R}{P + R}$$
$$\text{FPR} = \frac{FP}{FP + VN} \qquad \text{AUC-ROC} = \int_{0}^{1} \text{TPR}(FPR^{-1}(t)) \, dt$$

### Avaliação Estratificada por Categoria de Ataque
Como os detectores de anomalias (B1, B2, B3, B3b, B6a, B6b) são classificadores binários não-supervisionados, a avaliação multiclasse consiste na **Taxa de Detecção (Recall) Estratificada por Família de Ataque**:
$$\text{Recall}_k = \frac{\text{Ataques detectados da classe } k}{\text{Total de instâncias da classe } k \text{ no teste}}$$
Isso quantifica a cobertura do detector sobre ameaças específicas (ex.: DoS, PortScan, Brute Force, Web Attacks, Botnet, Infiltration). Métricas completas multiclasse de precisão e F1 por classe são reportadas exclusivamente para o Random Forest supervisionado.

---

## 4.8.4 Métodos de Decisão Híbrida (Framework Proposto - B6b)

O modelo final combina a representação latente comprimida do FC-DAE com a capacidade de isolamento do DIF através de duas formulações concorrentes:

### 1. Fusão Linear Ponderada com Normalização Robusta (Método Principal)
$$\text{Score}_{\text{final}}(x) = \alpha \cdot s_{\text{DIF}}(x) + (1 - \alpha) \cdot e_{\text{MSE, norm}}(x)$$
onde o erro de reconstrução é normalizado por percentis robustos calibrados na validação:
$$e_{\text{MSE, norm}}(x) = \text{clip}\left(\frac{e(x) - p_{1, \text{val}}}{p_{99, \text{val}} - p_{1, \text{val}}}, 0.0, 1.0\right)$$
A saturação intencional em 1.0 para ataques severos impede que outliers extremos desequilibrem a ponderação linear. O peso $\alpha \in [0, 1]$ é otimizado através de grid search na partição de validação para maximizar o Índice de Youden ($J = \text{Recall} - \text{FPR}$).

### 2. Regra Disjuntiva (Regra OR — Variante Avaliada)
Classifica uma amostra como maliciosa se violar qualquer um dos dois limiares calibrados na validação:
$$\hat{y}(x) = \mathbb{I}\left(s_{\text{DIF}}(x) > \tau_{\text{DIF, opt}} \lor e_{\text{MSE}}(x) > \tau_{\text{AE, opt}}\right)$$
Preserva a propriedade de "veto especializado", onde desvios estruturais densos são capturados pelo FC-DAE e desvios topológicos geométricos são capturados pelo DIF.

---

## 4.8.5 Baseline LSTM-AE e Requisito Temporal

O baseline B5 opera sobre janelas deslizantes cronológicas de tamanho fixo $W = 10$ fluxos com passo $S = 1$, ordenados pela coluna `Timestamp`.
*   **Decisão de Rotulação com Lookback:** O rótulo e o erro da janela são atribuídos ao fluxo final ($t_W$). Reconhece-se a assimetria metodológica de que a LSTM opera com contexto retrospectivo que os modelos de fluxo único não utilizam.
*   **Dependência de Pré-processamento:** A execução do baseline B5 requer preservar previamente a coluna temporal original dos fluxos em `timestamps.parquet` antes do descarte das colunas não-preditivas.

---

## 4.8.6 Protocolo de Análise de Sensibilidade via OFAT

A análise sistemática de sensibilidade adota a metodologia **One-Factor-at-a-Time (OFAT)** partindo da configuração baseline padrão:
$$\text{Configuração Padrão:} \quad m = \lfloor n/8 \rfloor = 9, \quad L = 3, \quad t = 100, \quad p = 95\%$$

Varreduras univariadas realizadas:
*   **Dimensão Latente ($m$):** $m \in \{9, 12, 19, 38\}$ (para $n=77$ atributos do CICIDS2017).  
    *Nota de Custo:* O valor padrão $m=9$ constitui o extremo inferior da varredura. Cada ponto avaliado de $m$ exige o retreinamento completo do FC-DAE.
*   **Profundidade das Redes $\Phi_i$ ($L$):** $L \in \{1, 2, 3, 4\}$ camadas densas.
*   **Número de Árvores ($t$):** $t \in \{50, 100, 200\}$ árvores de isolamento.
*   **Percentil do Limiar Não-Supervisionado ($p$):** $p \in \{90, 95, 97, 99\}\%$.

> [!NOTE]
> **Limitação Metodológica Reconhecida:**  
> O método OFAT reduz o espaço experimental de 192 para 15 execuções reprodutíveis, mas não mapeia interações de segunda ordem entre parâmetros (ex.: sensibilidade cruzada entre $L$ e $m$). Essa limitação é assumida explicitamente no TCC em favor da viabilidade computacional.

---

## 4.8.7 Checklist de Sincronização com o Texto do TCC

- [ ] **Seção 4.8.1:** Registrar que 20% das anomalias são reservadas para a validação (viabilizando Youden sem vazamento de teste), com nota de ressalva para classes ultra-raras.
- [ ] **Seção 4.8.2:** Inserir o **Baseline 3b (AE + DIF puro com $\lambda=0$)** para completar a cadeia formal de ablação do DEAS.
- [ ] **Seção 4.8.3:** Substituir a menção de "avaliação multiclasse" por "Taxa de Detecção (Recall) estratificada por família de ataque".
- [ ] **Seção 4.8.4:** Registrar a Normalização Robusta por percentis ($p_1$ e $p_{99}$) na Fusão Ponderada e a comparação com a Regra OR.
- [ ] **Seção 4.8.4 (Sensibilidade):** Declarar o método OFAT, apontar $m=9$ como extremo inferior e assumir a ausência de interações de segunda ordem.
- [ ] **Seção 4.8.2 (LSTM):** Documentar o formato de janela $W=10$ com atribuição em $t_W$ e a preservação de `timestamps.parquet`.

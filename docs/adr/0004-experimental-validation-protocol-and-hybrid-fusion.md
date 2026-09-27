# 4. Protocolo Experimental, Particionamento sem Vazamento e Fusão Híbrida

Data: 2026-09-21

## Status

Aceito

## Contexto

A Seção 4.8 do TCC define o protocolo experimental de validação do framework híbrido (FC-DAE + DIF), seus baselines comparativos, métricas e análise de sensibilidade. Fazia-se necessário resolver contradições metodológicas entre a alocação de amostras de ataque e a calibração de Youden, definir formalmente a arquitetura híbrida de decisão ("Threshold Adaptativo"), incluir baselines de ablação para comprovar o DEAS e precisar o significado da avaliação multiclasse para detectores de anomalias não-supervisionados.

## Decisão

1. **Particionamento sem Vazamento de Dados (*Data Leakage*):**
   - Tráfego Legítimo Normal: Particionado em 70% treino (ajuste não-supervisionado do FC-DAE e DIF), 15% validação (calibração de $\tau_{\text{ref}}$) e 15% teste cego final.
   - Tráfego Anômalo (Ataques): Particionado de forma **estratificada por classe** em 20% para a validação (viabilizando a construção da curva ROC e o cálculo legítimo do Índice de Youden sem contaminação do teste) e 80% para o teste cego final. Classes ultra-raras (ex.: Heartbleed com 11 instâncias no CICIDS2017) terão seu recall reportado com nota metodológica explícita de significância amostral.
2. **Arquitetura Híbrida e Normalização Robusta:**
   - **Fusão Linear Ponderada (Método Principal):** $\text{Score}_{\text{final}} = \alpha \cdot s_{\text{DIF}} + (1 - \alpha) \cdot e_{\text{MSE, norm}}$.
   - **Normalização Robusta do MSE:** Utiliza escalonamento Min-Max baseado nos percentis $p_1$ e $p_{99}$ da validação normal com *clipping* rígido em $[0, 1]$:
     $$e_{\text{MSE, norm}}(x) = \text{clip}\left(\frac{e(x) - p_{1, \text{val}}}{p_{99, \text{val}} - p_{1, \text{val}}}, 0.0, 1.0\right)$$
     A saturação em 1.0 é intencional para ataques severos, prevenindo que desvios extremos distorçam a ponderação. O peso $\alpha \in [0, 1]$ é otimizado na validação rotulada.
   - **Regra Disjuntiva (Regra OR — Variante Avaliada):** $\hat{y}(x) = \mathbb{I}(s_{\text{DIF}} > \tau_{\text{DIF, opt}} \lor e_{\text{MSE}} > \tau_{\text{AE, opt}})$. Avaliada em paralelo para demonstrar o poder de veto especializado de cada detector.
3. **Cadeia de Baselines com Três Degraus de Ablação:**
   - (B1) iForest clássico isolado sobre features brutas
   - (B2) FC-DAE isolado com erro MSE
   - (B3) Pre-IF com iForest clássico (FC-DAE + iForest sem projeções e sem DEAS)
   - **(B3b) Baseline de Ablação: Pre-IF + DIF puro ($\lambda = 0$, com projeções $\Phi_i$ e sem DEAS) — isola o ganho de $\Phi_i$**
   - (B4) Random Forest supervisionado (referência de teto de desempenho com rótulos)
   - (B5) LSTM Autoencoder (LSTM-AE) recorrente para modelagem temporal
   - **(B6a) Pre-IF + DIF com DEAS ($\lambda > 0$, isolado) — isola o ganho do DEAS na mitigação de ghost regions**
   - **(B6b) Arquitetura Proposta Completa (AE + DIF + DEAS + Fusão Ponderada / Regra OR)**
4. **Modelagem do Baseline LSTM-AE e Requisito Temporal:**
   O baseline recorrente B5 requer janelas deslizantes cronológicas de $W = 10$ fluxos ordenados por `Timestamp` com passo $S = 1$, atribuindo a predição ao fluxo final $t_W$. Reconhece-se a assimetria metodológica de que o LSTM possui contexto retrospectivo (*lookback*) que os detectores puramente pontuais não possuem. A execução de B5 requer a preservação prévia da coluna de timestamp dos fluxos em `timestamps.parquet`.
5. **Análise de Sensibilidade via OFAT (One-Factor-at-a-Time):**
   Adota-se a metodologia univariada OFAT partindo da baseline padrão ($m = \lfloor n/8 \rfloor = 9$, $L = 3$, $t = 100$, $p = 95\%$). Destaca-se que $m=9$ representa o extremo inferior da varredura ($m \in \{9, 12, 19, 38\}$ no CICIDS2017) e que variar $m$ acarreta o custo computacional de retreinar o FC-DAE para cada dimensão. Assume-se abertamente a limitação do OFAT de não mapear interações de segunda ordem entre hiperparâmetros, justificando a escolha pelo orçamento computacional reprodutível (15 execuções em vez de 192).
6. **Semântica da Avaliação por Categoria de Ataque:**
   Para detectores não-supervisionados, a avaliação multiclasse consiste estritamente na **Taxa de Detecção (Recall) Estratificada por Família de Ataque**. A precisão e o F1 multiclasse completos são reportados unicamente para o Random Forest supervisionado.

## Consequências

### Positivas
- Validação limpa e sem vazamento de dados (*data leakage*) com calibração legítima de Youden.
- Cadeia comparativa irrefutável comprovando separadamente as projeções $\Phi_i$, o DEAS e a fusão híbrida.
- Normalização robusta imune a outliers na validação.
- Orçamento computacional viável via OFAT.

### Negativas
- A varredura de $m$ exige retreinar o FC-DAE para cada valor candidato.
- O baseline LSTM-AE exige preservar metadados temporais originais antes do descarte de colunas no pré-processamento.

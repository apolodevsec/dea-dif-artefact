# Relatório Experimental — CICIDS2017

Protocolo *leakage-free*: o conjunto de teste fornecido no artefato permanece cego. Toda calibração (limiares, ponto de operação de Youden e alfa da fusão) consome exclusivamente partições derivadas de `X_train_baseline`.

## 1. Partições e Protocolo

| Partição | Composição | Fluxos |
| :--- | :--- | ---: |
| Treino FC-DAE + DIF | benigno (`BENIGN`) | 1,128,520 |
| Validação não-supervisionada | benigno | 199,150 |
| Validação rotulada | benigno + ataques estratificados | 246,259 (47,109 ataques) |
| Ataques reservados ao B4 | ataques de treino | 188,435 |
| **Teste cego** | benigno + ataques | 669,950 (100,948 ataques) |

- Features de entrada (n): **77** | coluna de rótulo: `Label`
- Divisão benigna treino/validação: 85% / 15% | ataques de treino movidos para validação: 20%
- Semente determinística: `42`

**Particionamento efetivo** (fração do total de cada classe, já que a fronteira treino/teste vem fixada pelo artefato):

| Classe | Treino não-supervisionado | Validação | Reservado ao B4 | Teste |
| :--- | :---: | :---: | :---: | :---: |
| Benigno (1,896,672) | 59.5% | 10.5% | — | 30.0% |
| Ataque (336,492) | 0% | 14.0% | 56.0% | 30.0% |

### 1.1 Procedência do Split Fornecido

| Evidência | Valor |
| :--- | :---: |
| Fração do teste sobre o total | 30.00% |
| Maior divergência de proporção de classe (treino vs. teste) | 0.0001 p.p. |
| Famílias exclusivas do teste | nenhuma |
| Cenário de ataques inéditos (*zero-day*) avaliado | **não** |

> [!WARNING]
> Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado.

> [!WARNING]
> **Baseline B5 (LSTM-AE) indisponível neste artefato.** Nenhuma feature temporal foi detectada e não existe arquivo paralelo de timestamps/metadados no diretório do dataset, de modo que o janelamento cronológico (W=10, passo S=1) não tem base. Reexecutar o Módulo 1 preservando a coluna `Timestamp` é pré-requisito para B5.

## 2. Treino e Validação do FC-DAE (Módulo 2)

| Item | Valor |
| :--- | :---: |
| Topologia | n=77 → n/2 → n/4 → m=9 → n/4 → n/2 → n |
| Melhor perda de validação (MSE) | `0.000214` |
| Melhor época | 60 de 60 executadas |
| Early stopping acionado | não |
| Duração do treino | 807.7 s |
| Dispositivo | `cpu` |

Limiar não-supervisionado do autoencoder, calibrado no MSE do tráfego benigno de validação. **Regra adotada: `percentile`** → `tau_AE = 0.000174`.

| Formulação | `tau_AE` | FPR esperado no benigno | Adotada |
| :--- | :---: | :---: | :---: |
| Percentil `p = 95` (Eq. 4 / Seção 4.7.1) | `0.000174` | 5.00% | **sim** |
| `mu + 3.0*sigma` (`ae.md`) | `0.004876` | 1.26% | não |

> [!IMPORTANT]
> As duas formulações **não são equivalentes**. `mu + k*sigma` só coincidiria com o percentil 99 se o MSE fosse aproximadamente normal; neste dataset ele cai no percentil empírico **98.74**. O texto do TCC e o código devem declarar a mesma regra — atualmente o padrão do pipeline é o percentil `p`, alinhado à Eq. 4 e à varredura OFAT do fator `p`.

## 3. Treino e Calibração do DIF (Módulo 4)

| Hiperparâmetro | Valor |
| :--- | :---: |
| Árvores de isolamento (t) | 100 |
| Subamostra por árvore (psi) | 256 |
| Redes de projeção (r) | 10 |
| Camadas por rede (L) | 3 |
| Dimensão de projeção (d) | 9 |
| Fator DEAS (lambda) | 0.5 |
| Duração do ajuste | 4.2 s |

Limiar de referência não-supervisionado no percentil 95.0 da validação benigna: `tau_DIF_ref(DEAS) = 0.627946`, `tau_DIF_ref(padrão) = 0.574753`.

## 4. Tabela 4 — Métricas Comparativas no Teste Cego

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 89.34% | 62.24% | 74.35% | 67.76% | 0.8768 | 8.00% | 0.0045 |
| **B2_Autoencoder_Alone** | 89.66% | 62.48% | 78.56% | 69.60% | 0.9051 | 8.37% | 0.0000 |
| **B3_PreIF_iForest** | 71.16% | 32.91% | 87.99% | 47.90% | 0.8482 | 31.83% | 0.0049 |
| **B3b_PreIF_DIF_Pure** | 72.33% | 33.38% | 83.99% | 47.77% | 0.8533 | 29.74% | 0.0455 |
| **B4_RandomForest_Supervised** | 99.85% | 99.70% | 99.33% | 99.51% | 0.9998 | 0.05% | 0.0014 |
| **B6a_Hybrid_Linear** | 89.66% | 62.48% | 78.56% | 69.60% | 0.9036 | 8.37% | 0.0455 |
| **B6b_Hybrid_OR** | 73.66% | 35.05% | 87.76% | 50.10% | 0.8794 | 28.85% | 0.0455 |

- **B1_iForest_Raw** — Isolation Forest clássico sobre as features brutas normalizadas.
- **B2_Autoencoder_Alone** — FC-DAE isolado, decidindo pelo erro de reconstrução MSE.
- **B3_PreIF_iForest** — Pre-IF: Isolation Forest clássico sobre o espaço latente z.
- **B3b_PreIF_DIF_Pure** — Ablação: DIF com lambda = 0 (projeções Phi_i sem DEAS).
- **B4_RandomForest_Supervised** — Teto comparativo supervisionado (Random Forest rotulado).
- **B6a_Hybrid_Linear** — Arquitetura proposta: DIF+DEAS fundido linearmente ao MSE (alfa calibrado).
- **B6b_Hybrid_OR** — Arquitetura proposta com regra de decisão disjuntiva (OR).

## 5. Ablação do DEAS

Cadeia comparativa de três degraus:
1. **Espaço latente vs. features brutas:** B1 → B3.
2. **Projeções neurais não-lineares Phi_i:** B3 → B3b (DIF com lambda = 0).
3. **Contribuição isolada do DEAS:** B3b → B6a (lambda > 0 + fusão linear).

- Ganho de F1-Score (Delta_DEAS_F1): `+0.2183`
- Ganho de AUC-ROC (Delta_DEAS_AUC): `+0.0503`

> [!NOTE]
> B3b e B6 compartilham a **mesma floresta** (a indução das árvores não depende de lambda). A topologia fica fixa e apenas o termo de desvio contínuo varia, o que torna o delta atribuível exclusivamente ao DEAS. Como h_deas e h_standard são acumulados na mesma travessia, a latência reportada para os dois é a do passo de inferência compartilhado.

## 6. Taxa de Detecção Estratificada por Família de Ataque

| Família de Ataque | Total no Teste | B1_iForest_Raw | B2_Autoencoder_Alone | B3_PreIF_iForest | B3b_PreIF_DIF_Pure | B4_RandomForest_Supervised | B6a_Hybrid_Linear | B6b_Hybrid_OR |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS Hulk** | 51,854 | 89.83% | 93.68% | 90.31% | 89.90% | 99.45% | 93.68% | 94.17% |
| **DDoS** | 38,404 | 63.19% | 63.61% | 84.90% | 79.90% | 99.90% | 63.61% | 85.35% |
| **DoS GoldenEye** | 3,086 | 61.67% | 91.87% | 95.20% | 85.16% | 99.32% | 91.87% | 92.64% |
| **FTP-Patator** | 1,779 | 0.00% | 0.06% | 69.36% | 31.65% | 99.83% | 0.06% | 31.59% |
| **DoS slowloris** | 1,616 | 55.88% | 64.11% | 99.63% | 65.90% | 99.69% | 64.11% | 66.21% |
| **DoS Slowhttptest** | 1,568 | 82.14% | 95.22% | 99.68% | 95.60% | 98.60% | 95.22% | 95.60% |
| **SSH-Patator** | 966 | 0.00% | 79.30% | 93.27% | 90.17% | 94.62% | 79.30% | 80.02% |
| **PortScan** | 587 | 9.37% | 19.25% | 86.88% | 38.16% | 96.93% | 19.25% | 22.49% |
| **Web Attack - Brute Force** | 441 | 4.99% | 4.99% | 90.70% | 90.02% | 95.24% | 4.99% | 5.67% |
| **Bot** | 431 | 2.32% | 2.32% | 9.05% | 9.28% | 58.24% | 2.32% | 8.58% |
| **Web Attack - XSS** | 196 | 2.55% | 2.55% | 95.41% | 95.92% | 93.37% | 2.55% | 3.06% |
| **Infiltration** ⚠️ | 11 | 90.91% | 100.00% | 100.00% | 100.00% | 0.00% | 100.00% | 100.00% |
| **Web Attack - Sql Injection** ⚠️ | 6 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Heartbleed** ⚠️ | 3 | 100.00% | 100.00% | 100.00% | 100.00% | 66.67% | 100.00% | 100.00% |

> [!WARNING]
> **Ressalva estatística de classes raras (⚠️):** `Heartbleed` (3 instâncias), `Infiltration` (11 instâncias), `Web Attack - Sql Injection` (6 instâncias). Com menos de 15 instâncias no teste, as taxas de detecção têm alta variância e devem ser lidas qualitativamente.

## 7. Parâmetros Calibrados (Anti-Leakage)

| Parâmetro | Valor | Regime |
| :--- | :---: | :--- |
| `tau_AE` (regra `percentile`) | `0.000174` | não-supervisionado (benigno de validação) |
| `tau_DIF_ref` (p95.0) | `0.627946` | não-supervisionado (benigno de validação) |
| `B1_iForest_Raw` | `0.510173` | supervisionado (Youden, validação rotulada) |
| `B2_Autoencoder_Alone` | `0.000099` | supervisionado (Youden, validação rotulada) |
| `B3_PreIF_iForest` | `0.431712` | supervisionado (Youden, validação rotulada) |
| `B3b_PreIF_DIF_Pure` | `0.453374` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_tau` | `0.007529` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_alpha` | `0.000000` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_dif` | `0.503599` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_ae` | `0.000099` | supervisionado (Youden, validação rotulada) |

## 8. Análise de Sensibilidade OFAT

### Parâmetro `L`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.471575 | 43.52% | 0.8295 | 87.73% | 38.22% |
| 2 | 0.471575 | 43.52% | 0.8295 | 87.73% | 38.22% |
| 3 | 0.503599 | 48.98% | 0.8546 | 81.25% | 26.71% |
| 4 | 0.481160 | 46.98% | 0.8499 | 87.91% | 33.06% |

### Parâmetro `t`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 0.483702 | 46.63% | 0.8533 | 87.61% | 33.38% |
| 100 | 0.503599 | 48.98% | 0.8546 | 81.25% | 26.71% |
| 200 | 0.494869 | 48.48% | 0.8518 | 81.65% | 27.53% |

### Parâmetro `p`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 90.0 | 0.580613 | 55.03% | 0.8546 | 59.45% | 10.04% |
| 95.0 | 0.627946 | 52.74% | 0.8546 | 46.00% | 5.05% |
| 97.0 | 0.677281 | 10.53% | 0.8546 | 6.51% | 3.04% |
| 99.0 | 0.722040 | 2.36% | 0.8546 | 1.26% | 1.00% |

### Parâmetro `lambda_deas`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.453374 | 47.77% | 0.8533 | 83.99% | 29.74% |
| 0.25 | 0.481347 | 48.82% | 0.8540 | 81.22% | 26.88% |
| 0.5 | 0.503599 | 48.98% | 0.8546 | 81.25% | 26.71% |
| 0.75 | 0.522388 | 48.55% | 0.8553 | 82.98% | 28.18% |
| 1.0 | 0.547327 | 48.95% | 0.8558 | 82.91% | 27.64% |

> [!NOTE]
> A metodologia OFAT conduz varreduras univariadas a partir da configuração central de baseline. Vantagem: evita a explosão combinatória da grade O(K^4) mantendo custo O(4 * K). Ressalva assumida: a análise não captura second_order_interactions (covariâncias entre parâmetros, como a interação entre profundidade de representação L e dimensão latente m). Adicionalmente, m=9 constitui o limite inferior teórico floor(n/8) para n=77 do CICIDS2017.

## Observações Metodológicas

- O conjunto de teste fornecido no artefato foi usado uma única vez, na avaliação final; nenhum limiar, alfa ou hiperparâmetro foi ajustado sobre ele.
- Os ataques da validação provêm de `X_train_baseline`, mantendo o teste cego, conforme o particionamento estratificado com ataques na validação.
- `tau_AE` adotado pela regra `percentile` (percentil p=95 do MSE benigno de validação, Eq. 4 / Seção 4.7.1). As duas formulações são reportadas lado a lado porque não são equivalentes: neste dataset `mu + 3*sigma` cai no percentil empírico 98.74, não em 99.
- `tau_AE` e `tau_DIF_ref` são limiares não-supervisionados (só tráfego benigno). Os limiares de Youden e o alfa da fusão são supervisionados e constituem o teto de operação calibrado na validação rotulada.
- O baseline B5 (LSTM-AE) **não** é executado. O janelamento cronológico (W=10, passo S=1, rótulo em t_W) exige a coluna `Timestamp`, e o artefato processado não a contém: nenhuma das features é temporal e não há arquivo paralelo de metadados/timestamps no diretório do dataset. Para destravar B5 é preciso reexecutar o Módulo 1 preservando a coluna temporal original.
- Procedência do split treino/teste do artefato: Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado. (teste = 30.00% do total; maior divergência de proporção de classe entre treino e teste = 0.0001 p.p.; famílias exclusivas do teste = nenhuma).
- Particionamento efetivo sobre o total de cada classe — benigno 59.5% treino / 10.5% validação / 30.0% teste; ataques 14.0% validação / 56.0% reservados ao B4 / 30.0% teste. Difere do esquema nominal 70/15/15 porque a fronteira treino/teste é fixada pelo artefato e o teste permanece intocado; reparticionar para 70/15/15 exigiria reslicing do teste.
- Consistência verificada: `X_train_autoencoder.parquet` tem exatamente o mesmo número de linhas do subconjunto benigno de `X_train_baseline.parquet`.

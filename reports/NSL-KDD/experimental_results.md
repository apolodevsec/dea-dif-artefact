# Relatório Experimental — NSL-KDD

Protocolo *leakage-free*: o conjunto de teste fornecido no artefato permanece cego. Toda calibração (limiares, ponto de operação de Youden e alfa da fusão) consome exclusivamente partições derivadas de `X_train_baseline`.

## 1. Partições e Protocolo

| Partição | Composição | Fluxos |
| :--- | :--- | ---: |
| Treino FC-DAE + DIF | benigno (`Normal`) | 45,795 |
| Validação não-supervisionada | benigno | 8,081 |
| Validação rotulada | benigno + ataques estratificados | 18,007 (9,926 ataques) |
| Ataques reservados ao B4 | ataques de treino | 39,702 |
| **Teste cego** | benigno + ataques | 44,360 (21,270 ataques) |

- Features de entrada (n): **115** | coluna de rótulo: `xAttack`
- Divisão benigna treino/validação: 85% / 15% | ataques de treino movidos para validação: 20%
- Semente determinística: `42`

**Particionamento efetivo** (fração do total de cada classe, já que a fronteira treino/teste vem fixada pelo artefato):

| Classe | Treino não-supervisionado | Validação | Reservado ao B4 | Teste |
| :--- | :---: | :---: | :---: | :---: |
| Benigno (76,966) | 59.5% | 10.5% | — | 30.0% |
| Ataque (70,898) | 0% | 14.0% | 56.0% | 30.0% |

### 1.1 Procedência do Split Fornecido

| Evidência | Valor |
| :--- | :---: |
| Fração do teste sobre o total | 30.00% |
| Maior divergência de proporção de classe (treino vs. teste) | 0.0009 p.p. |
| Famílias exclusivas do teste | nenhuma |
| Cenário de ataques inéditos (*zero-day*) avaliado | **não** |

> [!WARNING]
> Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado.

> [!WARNING]
> **Baseline B5 (LSTM-AE) indisponível neste artefato.** Nenhuma feature temporal foi detectada e não existe arquivo paralelo de timestamps/metadados no diretório do dataset, de modo que o janelamento cronológico (W=10, passo S=1) não tem base. Reexecutar o Módulo 1 preservando a coluna `Timestamp` é pré-requisito para B5.

## 2. Treino e Validação do FC-DAE (Módulo 2)

| Item | Valor |
| :--- | :---: |
| Topologia | n=115 → n/2 → n/4 → m=14 → n/4 → n/2 → n |
| Melhor perda de validação (MSE) | `0.003684` |
| Melhor época | 60 de 60 executadas |
| Early stopping acionado | não |
| Duração do treino | 34.9 s |
| Dispositivo | `cpu` |

Limiar não-supervisionado do autoencoder, calibrado no MSE do tráfego benigno de validação. **Regra adotada: `percentile`** → `tau_AE = 0.023966`.

| Formulação | `tau_AE` | FPR esperado no benigno | Adotada |
| :--- | :---: | :---: | :---: |
| Percentil `p = 95` (Eq. 4 / Seção 4.7.1) | `0.023966` | 5.00% | **sim** |
| `mu + 3.0*sigma` (`ae.md`) | `0.031035` | 4.27% | não |

> [!IMPORTANT]
> As duas formulações **não são equivalentes**. `mu + k*sigma` só coincidiria com o percentil 99 se o MSE fosse aproximadamente normal; neste dataset ele cai no percentil empírico **95.73**. O texto do TCC e o código devem declarar a mesma regra — atualmente o padrão do pipeline é o percentil `p`, alinhado à Eq. 4 e à varredura OFAT do fator `p`.

## 3. Treino e Calibração do DIF (Módulo 4)

| Hiperparâmetro | Valor |
| :--- | :---: |
| Árvores de isolamento (t) | 100 |
| Subamostra por árvore (psi) | 256 |
| Redes de projeção (r) | 10 |
| Camadas por rede (L) | 3 |
| Dimensão de projeção (d) | 14 |
| Fator DEAS (lambda) | 0.5 |
| Duração do ajuste | 0.8 s |

Limiar de referência não-supervisionado no percentil 95.0 da validação benigna: `tau_DIF_ref(DEAS) = 0.649353`, `tau_DIF_ref(padrão) = 0.592017`.

## 4. Tabela 4 — Métricas Comparativas no Teste Cego

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 90.42% | 92.76% | 86.80% | 89.68% | 0.9701 | 6.24% | 0.0057 |
| **B2_Autoencoder_Alone** | 91.54% | 91.09% | 91.29% | 91.19% | 0.9765 | 8.22% | 0.0000 |
| **B3_PreIF_iForest** | 85.14% | 80.89% | 90.37% | 85.37% | 0.9094 | 19.67% | 0.0054 |
| **B3b_PreIF_DIF_Pure** | 85.69% | 80.15% | 93.25% | 86.20% | 0.9021 | 21.27% | 0.0792 |
| **B4_RandomForest_Supervised** | 99.56% | 99.72% | 99.36% | 99.54% | 0.9998 | 0.26% | 0.0025 |
| **B6a_Hybrid_Linear** | 92.67% | 91.74% | 93.09% | 92.41% | 0.9703 | 7.72% | 0.0792 |
| **B6b_Hybrid_OR** | 85.97% | 78.74% | 96.91% | 86.88% | 0.9669 | 24.11% | 0.0792 |

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

- Ganho de F1-Score (Delta_DEAS_F1): `+0.0621`
- Ganho de AUC-ROC (Delta_DEAS_AUC): `+0.0682`

> [!NOTE]
> B3b e B6 compartilham a **mesma floresta** (a indução das árvores não depende de lambda). A topologia fica fixa e apenas o termo de desvio contínuo varia, o que torna o delta atribuível exclusivamente ao DEAS. Como h_deas e h_standard são acumulados na mesma travessia, a latência reportada para os dois é a do passo de inferência compartilhado.

## 6. Taxa de Detecção Estratificada por Família de Ataque

| Família de Ataque | Total no Teste | B1_iForest_Raw | B2_Autoencoder_Alone | B3_PreIF_iForest | B3b_PreIF_DIF_Pure | B4_RandomForest_Supervised | B6a_Hybrid_Linear | B6b_Hybrid_OR |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DoS** | 15,890 | 92.28% | 96.79% | 90.94% | 94.07% | 99.91% | 96.95% | 97.78% |
| **Probe** | 4,179 | 87.56% | 88.20% | 97.54% | 98.78% | 99.69% | 96.63% | 99.95% |
| **R2L** | 1,125 | 8.62% | 25.87% | 56.71% | 62.22% | 92.09% | 26.67% | 74.13% |
| **U2R** | 76 | 56.58% | 80.26% | 75.00% | 76.32% | 72.37% | 75.00% | 85.53% |

## 7. Parâmetros Calibrados (Anti-Leakage)

| Parâmetro | Valor | Regime |
| :--- | :---: | :--- |
| `tau_AE` (regra `percentile`) | `0.023966` | não-supervisionado (benigno de validação) |
| `tau_DIF_ref` (p95.0) | `0.649353` | não-supervisionado (benigno de validação) |
| `B1_iForest_Raw` | `0.484057` | supervisionado (Youden, validação rotulada) |
| `B2_Autoencoder_Alone` | `0.012295` | supervisionado (Youden, validação rotulada) |
| `B3_PreIF_iForest` | `0.523446` | supervisionado (Youden, validação rotulada) |
| `B3b_PreIF_DIF_Pure` | `0.515713` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_tau` | `0.508990` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_alpha` | `0.720000` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_dif` | `0.562155` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_ae` | `0.012295` | supervisionado (Youden, validação rotulada) |

## 8. Análise de Sensibilidade OFAT

### Parâmetro `L`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.562843 | 86.84% | 0.9087 | 94.89% | 21.78% |
| 2 | 0.562843 | 86.84% | 0.9087 | 94.89% | 21.78% |
| 3 | 0.562155 | 85.78% | 0.8998 | 93.24% | 22.25% |
| 4 | 0.577804 | 85.63% | 0.9229 | 89.65% | 18.19% |

### Parâmetro `t`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 0.554017 | 85.45% | 0.8959 | 95.46% | 25.75% |
| 100 | 0.562155 | 85.78% | 0.8998 | 93.24% | 22.25% |
| 200 | 0.568195 | 85.29% | 0.8982 | 90.84% | 20.42% |

### Parâmetro `p`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 90.0 | 0.621604 | 60.85% | 0.8998 | 48.71% | 10.49% |
| 95.0 | 0.649353 | 58.45% | 0.8998 | 43.70% | 5.37% |
| 97.0 | 0.663173 | 47.50% | 0.8998 | 32.25% | 3.24% |
| 99.0 | 0.692951 | 30.58% | 0.8998 | 18.24% | 0.99% |

### Parâmetro `lambda_deas`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.515713 | 86.20% | 0.9021 | 93.25% | 21.27% |
| 0.25 | 0.538228 | 86.03% | 0.9010 | 93.37% | 21.83% |
| 0.5 | 0.562155 | 85.78% | 0.8998 | 93.24% | 22.25% |
| 0.75 | 0.585551 | 85.53% | 0.8985 | 93.65% | 23.33% |
| 1.0 | 0.610728 | 85.36% | 0.8969 | 93.81% | 23.94% |

> [!NOTE]
> A metodologia OFAT conduz varreduras univariadas a partir da configuração central de baseline. Vantagem: evita a explosão combinatória da grade O(K^4) mantendo custo O(4 * K). Ressalva assumida: a análise não captura second_order_interactions (covariâncias entre parâmetros, como a interação entre profundidade de representação L e dimensão latente m). Adicionalmente, m=9 constitui o limite inferior teórico floor(n/8) para n=77 do CICIDS2017.

## Observações Metodológicas

- O conjunto de teste fornecido no artefato foi usado uma única vez, na avaliação final; nenhum limiar, alfa ou hiperparâmetro foi ajustado sobre ele.
- Os ataques da validação provêm de `X_train_baseline`, mantendo o teste cego, conforme o particionamento estratificado com ataques na validação.
- `tau_AE` adotado pela regra `percentile` (percentil p=95 do MSE benigno de validação, Eq. 4 / Seção 4.7.1). As duas formulações são reportadas lado a lado porque não são equivalentes: neste dataset `mu + 3*sigma` cai no percentil empírico 95.73, não em 99.
- `tau_AE` e `tau_DIF_ref` são limiares não-supervisionados (só tráfego benigno). Os limiares de Youden e o alfa da fusão são supervisionados e constituem o teto de operação calibrado na validação rotulada.
- O baseline B5 (LSTM-AE) **não** é executado. O janelamento cronológico (W=10, passo S=1, rótulo em t_W) exige a coluna `Timestamp`, e o artefato processado não a contém: nenhuma das features é temporal e não há arquivo paralelo de metadados/timestamps no diretório do dataset. Para destravar B5 é preciso reexecutar o Módulo 1 preservando a coluna temporal original.
- Procedência do split treino/teste do artefato: Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado. (teste = 30.00% do total; maior divergência de proporção de classe entre treino e teste = 0.0009 p.p.; famílias exclusivas do teste = nenhuma).
- Particionamento efetivo sobre o total de cada classe — benigno 59.5% treino / 10.5% validação / 30.0% teste; ataques 14.0% validação / 56.0% reservados ao B4 / 30.0% teste. Difere do esquema nominal 70/15/15 porque a fronteira treino/teste é fixada pelo artefato e o teste permanece intocado; reparticionar para 70/15/15 exigiria reslicing do teste.
- Consistência verificada: `X_train_autoencoder.parquet` tem exatamente o mesmo número de linhas do subconjunto benigno de `X_train_baseline.parquet`.

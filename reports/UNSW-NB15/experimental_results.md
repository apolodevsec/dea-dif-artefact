# Relatório Experimental — UNSW-NB15

Protocolo *leakage-free*: o conjunto de teste fornecido no artefato permanece cego. Toda calibração (limiares, ponto de operação de Youden e alfa da fusão) consome exclusivamente partições derivadas de `X_train_baseline`.

## 1. Partições e Protocolo

| Partição | Composição | Fluxos |
| :--- | :--- | ---: |
| Treino FC-DAE + DIF | benigno (`Normal`) | 51,004 |
| Validação não-supervisionada | benigno | 9,001 |
| Validação rotulada | benigno + ataques estratificados | 19,188 (10,187 ataques) |
| Ataques reservados ao B4 | ataques de treino | 40,746 |
| **Teste cego** | benigno + ataques | 47,545 (21,828 ataques) |

- Features de entrada (n): **194** | coluna de rótulo: `attack_cat`
- Divisão benigna treino/validação: 85% / 15% | ataques de treino movidos para validação: 20%
- Semente determinística: `42`

**Particionamento efetivo** (fração do total de cada classe, já que a fronteira treino/teste vem fixada pelo artefato):

| Classe | Treino não-supervisionado | Validação | Reservado ao B4 | Teste |
| :--- | :---: | :---: | :---: | :---: |
| Benigno (85,722) | 59.5% | 10.5% | — | 30.0% |
| Ataque (72,761) | 0% | 14.0% | 56.0% | 30.0% |

### 1.1 Procedência do Split Fornecido

| Evidência | Valor |
| :--- | :---: |
| Fração do teste sobre o total | 30.00% |
| Maior divergência de proporção de classe (treino vs. teste) | 0.0015 p.p. |
| Famílias exclusivas do teste | nenhuma |
| Cenário de ataques inéditos (*zero-day*) avaliado | **não** |

> [!WARNING]
> Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado.

> [!WARNING]
> **Baseline B5 (LSTM-AE) indisponível neste artefato.** Nenhuma feature temporal foi detectada e não existe arquivo paralelo de timestamps/metadados no diretório do dataset, de modo que o janelamento cronológico (W=10, passo S=1) não tem base. Reexecutar o Módulo 1 preservando a coluna `Timestamp` é pré-requisito para B5.

## 2. Treino e Validação do FC-DAE (Módulo 2)

| Item | Valor |
| :--- | :---: |
| Topologia | n=194 → n/2 → n/4 → m=24 → n/4 → n/2 → n |
| Melhor perda de validação (MSE) | `0.003292` |
| Melhor época | 58 de 60 executadas |
| Early stopping acionado | não |
| Duração do treino | 42.1 s |
| Dispositivo | `cpu` |

Limiar não-supervisionado do autoencoder, calibrado no MSE do tráfego benigno de validação. **Regra adotada: `percentile`** → `tau_AE = 0.012036`.

| Formulação | `tau_AE` | FPR esperado no benigno | Adotada |
| :--- | :---: | :---: | :---: |
| Percentil `p = 95` (Eq. 4 / Seção 4.7.1) | `0.012036` | 5.00% | **sim** |
| `mu + 3.0*sigma` (`ae.md`) | `0.015563` | 1.50% | não |

> [!IMPORTANT]
> As duas formulações **não são equivalentes**. `mu + k*sigma` só coincidiria com o percentil 99 se o MSE fosse aproximadamente normal; neste dataset ele cai no percentil empírico **98.50**. O texto do TCC e o código devem declarar a mesma regra — atualmente o padrão do pipeline é o percentil `p`, alinhado à Eq. 4 e à varredura OFAT do fator `p`.

## 3. Treino e Calibração do DIF (Módulo 4)

| Hiperparâmetro | Valor |
| :--- | :---: |
| Árvores de isolamento (t) | 100 |
| Subamostra por árvore (psi) | 256 |
| Redes de projeção (r) | 10 |
| Camadas por rede (L) | 3 |
| Dimensão de projeção (d) | 24 |
| Fator DEAS (lambda) | 0.5 |
| Duração do ajuste | 0.8 s |

Limiar de referência não-supervisionado no percentil 95.0 da validação benigna: `tau_DIF_ref(DEAS) = 0.634876`, `tau_DIF_ref(padrão) = 0.578805`.

## 4. Tabela 4 — Métricas Comparativas no Teste Cego

| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1_iForest_Raw** | 67.40% | 64.66% | 63.92% | 64.29% | 0.6949 | 29.65% | 0.0084 |
| **B2_Autoencoder_Alone** | 68.28% | 64.95% | 67.13% | 66.02% | 0.7196 | 30.75% | 0.0000 |
| **B3_PreIF_iForest** | 68.35% | 65.06% | 67.08% | 66.05% | 0.7473 | 30.58% | 0.0055 |
| **B3b_PreIF_DIF_Pure** | 68.16% | 60.60% | 87.59% | 71.64% | 0.7737 | 48.33% | 0.0744 |
| **B4_RandomForest_Supervised** | 91.81% | 92.02% | 89.97% | 90.98% | 0.9807 | 6.63% | 0.0034 |
| **B6a_Hybrid_Linear** | 72.13% | 70.41% | 67.79% | 69.08% | 0.7712 | 24.18% | 0.0744 |
| **B6b_Hybrid_OR** | 62.64% | 55.88% | 88.44% | 68.49% | 0.7623 | 59.27% | 0.0744 |

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

- Ganho de F1-Score (Delta_DEAS_F1): `-0.0256`
- Ganho de AUC-ROC (Delta_DEAS_AUC): `-0.0025`

> [!NOTE]
> B3b e B6 compartilham a **mesma floresta** (a indução das árvores não depende de lambda). A topologia fica fixa e apenas o termo de desvio contínuo varia, o que torna o delta atribuível exclusivamente ao DEAS. Como h_deas e h_standard são acumulados na mesma travessia, a latência reportada para os dois é a do passo de inferência compartilhado.

## 6. Taxa de Detecção Estratificada por Família de Ataque

| Família de Ataque | Total no Teste | B1_iForest_Raw | B2_Autoencoder_Alone | B3_PreIF_iForest | B3b_PreIF_DIF_Pure | B4_RandomForest_Supervised | B6a_Hybrid_Linear | B6b_Hybrid_OR |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exploits** | 7,787 | 70.01% | 78.79% | 65.83% | 90.95% | 98.10% | 76.28% | 92.00% |
| **Fuzzers** | 5,933 | 46.10% | 40.72% | 55.49% | 79.69% | 67.64% | 46.22% | 80.14% |
| **Reconnaissance** | 2,839 | 46.07% | 52.48% | 64.56% | 82.00% | 99.79% | 56.18% | 82.74% |
| **Generic** | 2,103 | 94.58% | 95.72% | 95.53% | 98.15% | 99.90% | 95.96% | 98.34% |
| **DoS** | 1,582 | 76.74% | 83.31% | 70.67% | 92.86% | 98.86% | 78.45% | 94.37% |
| **Analysis** | 581 | 96.56% | 97.42% | 84.85% | 96.90% | 86.92% | 91.91% | 98.80% |
| **Backdoor** | 541 | 84.84% | 87.80% | 87.43% | 93.72% | 99.82% | 87.99% | 95.75% |
| **Shellcode** | 413 | 48.43% | 47.46% | 61.50% | 80.39% | 95.88% | 50.36% | 81.36% |
| **Worms** | 49 | 71.43% | 91.84% | 89.80% | 95.92% | 97.96% | 89.80% | 97.96% |

## 7. Parâmetros Calibrados (Anti-Leakage)

| Parâmetro | Valor | Regime |
| :--- | :---: | :--- |
| `tau_AE` (regra `percentile`) | `0.012036` | não-supervisionado (benigno de validação) |
| `tau_DIF_ref` (p95.0) | `0.634876` | não-supervisionado (benigno de validação) |
| `B1_iForest_Raw` | `0.423626` | supervisionado (Youden, validação rotulada) |
| `B2_Autoencoder_Alone` | `0.005369` | supervisionado (Youden, validação rotulada) |
| `B3_PreIF_iForest` | `0.497943` | supervisionado (Youden, validação rotulada) |
| `B3b_PreIF_DIF_Pure` | `0.476371` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_tau` | `0.479119` | supervisionado (Youden, validação rotulada) |
| `B6a_Hybrid_Linear_alpha` | `0.770000` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_dif` | `0.522446` | supervisionado (Youden, validação rotulada) |
| `B6b_Hybrid_OR_tau_ae` | `0.005369` | supervisionado (Youden, validação rotulada) |

## 8. Análise de Sensibilidade OFAT

### Parâmetro `L`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.546506 | 68.71% | 0.7684 | 69.63% | 28.05% |
| 2 | 0.546506 | 68.71% | 0.7684 | 69.63% | 28.05% |
| 3 | 0.522446 | 71.28% | 0.7697 | 87.52% | 49.26% |
| 4 | 0.547063 | 70.21% | 0.7799 | 73.90% | 31.07% |

### Parâmetro `t`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 50 | 0.522895 | 67.82% | 0.7469 | 79.08% | 45.95% |
| 100 | 0.522446 | 71.28% | 0.7697 | 87.52% | 49.26% |
| 200 | 0.565564 | 65.38% | 0.7673 | 62.40% | 24.18% |

### Parâmetro `p`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 90.0 | 0.606102 | 42.36% | 0.7697 | 29.90% | 9.57% |
| 95.0 | 0.634876 | 21.40% | 0.7697 | 12.61% | 4.44% |
| 97.0 | 0.648326 | 18.62% | 0.7697 | 10.59% | 2.67% |
| 99.0 | 0.679184 | 4.12% | 0.7697 | 2.13% | 0.95% |

### Parâmetro `lambda_deas`

| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.0 | 0.476371 | 71.64% | 0.7737 | 87.59% | 48.33% |
| 0.25 | 0.499037 | 71.43% | 0.7718 | 87.43% | 48.70% |
| 0.5 | 0.522446 | 71.28% | 0.7697 | 87.52% | 49.26% |
| 0.75 | 0.548057 | 70.93% | 0.7675 | 86.78% | 49.16% |
| 1.0 | 0.571615 | 71.00% | 0.7652 | 88.41% | 51.46% |

> [!NOTE]
> A metodologia OFAT conduz varreduras univariadas a partir da configuração central de baseline. Vantagem: evita a explosão combinatória da grade O(K^4) mantendo custo O(4 * K). Ressalva assumida: a análise não captura second_order_interactions (covariâncias entre parâmetros, como a interação entre profundidade de representação L e dimensão latente m). Adicionalmente, m=9 constitui o limite inferior teórico floor(n/8) para n=77 do CICIDS2017.

## Observações Metodológicas

- O conjunto de teste fornecido no artefato foi usado uma única vez, na avaliação final; nenhum limiar, alfa ou hiperparâmetro foi ajustado sobre ele.
- Os ataques da validação provêm de `X_train_baseline`, mantendo o teste cego, conforme o particionamento estratificado com ataques na validação.
- `tau_AE` adotado pela regra `percentile` (percentil p=95 do MSE benigno de validação, Eq. 4 / Seção 4.7.1). As duas formulações são reportadas lado a lado porque não são equivalentes: neste dataset `mu + 3*sigma` cai no percentil empírico 98.50, não em 99.
- `tau_AE` e `tau_DIF_ref` são limiares não-supervisionados (só tráfego benigno). Os limiares de Youden e o alfa da fusão são supervisionados e constituem o teto de operação calibrado na validação rotulada.
- O baseline B5 (LSTM-AE) **não** é executado. O janelamento cronológico (W=10, passo S=1, rótulo em t_W) exige a coluna `Timestamp`, e o artefato processado não a contém: nenhuma das features é temporal e não há arquivo paralelo de metadados/timestamps no diretório do dataset. Para destravar B5 é preciso reexecutar o Módulo 1 preservando a coluna temporal original.
- Procedência do split treino/teste do artefato: Refatiamento aleatório estratificado: as proporções de classe do teste replicam as do treino e nenhuma família é exclusiva do teste. O split canônico do dataset (e com ele o cenário zero-day de ataques inéditos) NÃO está preservado. (teste = 30.00% do total; maior divergência de proporção de classe entre treino e teste = 0.0015 p.p.; famílias exclusivas do teste = nenhuma).
- Particionamento efetivo sobre o total de cada classe — benigno 59.5% treino / 10.5% validação / 30.0% teste; ataques 14.0% validação / 56.0% reservados ao B4 / 30.0% teste. Difere do esquema nominal 70/15/15 porque a fronteira treino/teste é fixada pelo artefato e o teste permanece intocado; reparticionar para 70/15/15 exigiria reslicing do teste.
- Consistência verificada: `X_train_autoencoder.parquet` tem exatamente o mesmo número de linhas do subconjunto benigno de `X_train_baseline.parquet`.

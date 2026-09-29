# Assets para o TCC — Capítulos 4 e 5

Gerado por `python -m src.make_tcc_assets` a partir dos relatórios em `reports/`.
Não editar à mão: regenerar após cada execução da rotina de treino.

Datasets incluídos: CICIDS2017, NSL-KDD, UNSW-NB15.

## Como usar

- **LaTeX:** `\input{tabelas/tabela4_NSL-KDD.tex}` e `\includegraphics[width=\linewidth]{figuras/curvas_roc_NSL-KDD.png}`.
- **Word / Google Docs:** cole o conteúdo do `.md` correspondente e insira o `.png`.
- As figuras estão a 300 dpi, em modo claro, e identificam cada série por cor **e** marcador/hachura, de modo a permanecerem legíveis em impressão monocromática.

## Capítulo 4 — Metodologia e Framework Proposto

| Arquivo | Seção | Conteúdo |
| :--- | :--- | :--- |
| `figuras/convergencia_fc_dae_CICIDS2017.png` | 4.7 — FC-DAE | Curva de convergência treino/validação (CICIDS2017) |
| `figuras/convergencia_fc_dae_NSL-KDD.png` | 4.7 — FC-DAE | Curva de convergência treino/validação (NSL-KDD) |
| `figuras/convergencia_fc_dae_UNSW-NB15.png` | 4.7 — FC-DAE | Curva de convergência treino/validação (UNSW-NB15) |
| `figuras/reconstrucao_diferencial_CICIDS2017.png` | 4.7.1 — Limiar de anomalia | Distribuição do MSE benigno vs. ataque com tau_AE (CICIDS2017) |
| `figuras/reconstrucao_diferencial_NSL-KDD.png` | 4.7.1 — Limiar de anomalia | Distribuição do MSE benigno vs. ataque com tau_AE (NSL-KDD) |
| `figuras/reconstrucao_diferencial_UNSW-NB15.png` | 4.7.1 — Limiar de anomalia | Distribuição do MSE benigno vs. ataque com tau_AE (UNSW-NB15) |
| `tabelas/particionamento.md` | 4.8.1 — Particionamento | Protocolo de particionamento efetivo por dataset (`.md`) |
| `tabelas/particionamento.tex` | 4.8.1 — Particionamento | Protocolo de particionamento efetivo por dataset (`.tex`) |
| `tabelas/calibracao_CICIDS2017.md` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (CICIDS2017) (`.md`) |
| `tabelas/calibracao_CICIDS2017.tex` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (CICIDS2017) (`.tex`) |
| `tabelas/calibracao_NSL-KDD.md` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (NSL-KDD) (`.md`) |
| `tabelas/calibracao_NSL-KDD.tex` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (NSL-KDD) (`.tex`) |
| `tabelas/calibracao_UNSW-NB15.md` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (UNSW-NB15) (`.md`) |
| `tabelas/calibracao_UNSW-NB15.tex` | 4.8.4 — Calibração | Limiares e alfa calibrados sem vazamento (UNSW-NB15) (`.tex`) |
| `figuras/ofat_L_CICIDS2017.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator L (CICIDS2017) |
| `figuras/ofat_L_NSL-KDD.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator L (NSL-KDD) |
| `figuras/ofat_L_UNSW-NB15.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator L (UNSW-NB15) |
| `figuras/ofat_lambda_deas_CICIDS2017.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator lambda_deas (CICIDS2017) |
| `figuras/ofat_lambda_deas_NSL-KDD.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator lambda_deas (NSL-KDD) |
| `figuras/ofat_lambda_deas_UNSW-NB15.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator lambda_deas (UNSW-NB15) |
| `figuras/ofat_p_CICIDS2017.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator p (CICIDS2017) |
| `figuras/ofat_p_NSL-KDD.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator p (NSL-KDD) |
| `figuras/ofat_p_UNSW-NB15.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator p (UNSW-NB15) |
| `figuras/ofat_t_CICIDS2017.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator t (CICIDS2017) |
| `figuras/ofat_t_NSL-KDD.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator t (NSL-KDD) |
| `figuras/ofat_t_UNSW-NB15.png` | 4.8.6 — Sensibilidade OFAT | Varredura univariada do fator t (UNSW-NB15) |

## Capítulo 5 — Resultados Experimentais

| Arquivo | Seção | Conteúdo |
| :--- | :--- | :--- |
| `figuras/ablacao_deas_f1.png` | 5.x — Ablação do DEAS | Ganho de F1 do DEAS por dataset (λ=0 vs. DEAS, mesma floresta) |
| `tabelas/ablacao_deas.md` | 5.x — Ablação do DEAS | Ganho isolado do DEAS em F1 e AUC por dataset (`.md`) |
| `tabelas/ablacao_deas.tex` | 5.x — Ablação do DEAS | Ganho isolado do DEAS em F1 e AUC por dataset (`.tex`) |
| `figuras/comparativo_auc.png` | 5.x — Comparação multi-dataset | auc_roc de cada baseline nas bases avaliadas |
| `figuras/comparativo_f1.png` | 5.x — Comparação multi-dataset | f1_score de cada baseline nas bases avaliadas |
| `figuras/curvas_roc_CICIDS2017.png` | 5.x — Desempenho comparativo | Curvas ROC da cadeia de ablação, com teto supervisionado (CICIDS2017) |
| `figuras/curvas_roc_NSL-KDD.png` | 5.x — Desempenho comparativo | Curvas ROC da cadeia de ablação, com teto supervisionado (NSL-KDD) |
| `figuras/curvas_roc_UNSW-NB15.png` | 5.x — Desempenho comparativo | Curvas ROC da cadeia de ablação, com teto supervisionado (UNSW-NB15) |
| `figuras/recall_estratificado_CICIDS2017.png` | 5.x — Detecção por família | Mapa de calor da taxa de detecção por família de ataque (CICIDS2017) |
| `figuras/recall_estratificado_NSL-KDD.png` | 5.x — Detecção por família | Mapa de calor da taxa de detecção por família de ataque (NSL-KDD) |
| `figuras/recall_estratificado_UNSW-NB15.png` | 5.x — Detecção por família | Mapa de calor da taxa de detecção por família de ataque (UNSW-NB15) |
| `tabelas/recall_estratificado_CICIDS2017.md` | 5.x — Detecção por família | Recall estratificado por família de ataque (CICIDS2017) (`.md`) |
| `tabelas/recall_estratificado_CICIDS2017.tex` | 5.x — Detecção por família | Recall estratificado por família de ataque (CICIDS2017) (`.tex`) |
| `tabelas/recall_estratificado_NSL-KDD.md` | 5.x — Detecção por família | Recall estratificado por família de ataque (NSL-KDD) (`.md`) |
| `tabelas/recall_estratificado_NSL-KDD.tex` | 5.x — Detecção por família | Recall estratificado por família de ataque (NSL-KDD) (`.tex`) |
| `tabelas/recall_estratificado_UNSW-NB15.md` | 5.x — Detecção por família | Recall estratificado por família de ataque (UNSW-NB15) (`.md`) |
| `tabelas/recall_estratificado_UNSW-NB15.tex` | 5.x — Detecção por família | Recall estratificado por família de ataque (UNSW-NB15) (`.tex`) |
| `tabelas/tabela4_CICIDS2017.md` | Tabela 4 | Métricas comparativas no teste cego (CICIDS2017) (`.md`) |
| `tabelas/tabela4_CICIDS2017.tex` | Tabela 4 | Métricas comparativas no teste cego (CICIDS2017) (`.tex`) |
| `tabelas/tabela4_NSL-KDD.md` | Tabela 4 | Métricas comparativas no teste cego (NSL-KDD) (`.md`) |
| `tabelas/tabela4_NSL-KDD.tex` | Tabela 4 | Métricas comparativas no teste cego (NSL-KDD) (`.tex`) |
| `tabelas/tabela4_UNSW-NB15.md` | Tabela 4 | Métricas comparativas no teste cego (UNSW-NB15) (`.md`) |
| `tabelas/tabela4_UNSW-NB15.tex` | Tabela 4 | Métricas comparativas no teste cego (UNSW-NB15) (`.tex`) |

## Atenção antes de escrever os resultados

> **Fusão degenerada.**

- **CICIDS2017**: `alpha = 0` — B6a colapsa em `B2_Autoencoder_Alone` (métricas idênticas). O `Delta_DEAS` deste dataset **não** é o ganho do DEAS e não deve ser reportado como tal.

## Ressalvas metodológicas a transportar para o texto

- Os splits fornecidos são refatiamentos estratificados 70/30, não os splits canônicos das bases: o cenário *zero-day* não está sendo avaliado.
- `tau_AE` usa o percentil `p=95`; `ae.md` e `CONTEXT.md` ainda descrevem "mu+3sigma ou percentil 99" e precisam ser alinhados.
- O baseline B5 (LSTM-AE) não foi executado: os artefatos não preservam `Timestamp`.
- Famílias com menos de 15 instâncias no teste aparecem marcadas; sua taxa de detecção tem alta variância e deve ser lida qualitativamente.

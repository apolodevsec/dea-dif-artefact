# 05: Pipeline CLI Desacoplado, Metadados e Persistência de Scores

**What to build:** Scripts executáveis de terminal `src/train_dif.py` e `src/score_dif.py`. O primeiro ajusta o modelo DIF estritamente sobre tráfego normal extraído do parquet latente, calcula o limiar não-supervisionado de referência $\tau_{\text{DIF, ref}}$ (percentil sobre validação) e serializa o modelo em `models/dif_ensemble.joblib`. O segundo carrega o modelo, realiza inferência em larga escala, registra telemetria de hardware e persiste `data/processed/dif_scores.parquet` contendo ambas as pontuações e rótulos originais.

**Blocked by:** 03-ghost-regions-and-outlier-discrimination, 04-representation-decoupling-and-vectorized-inference

**Status:** closed

- [x] Script CLI `src/train_dif.py` com argumentos `--latent_parquet`, `--model_out`, `--trees`, `--subsample`, `--representations`, `--percentile`.
- [x] Filtragem estrita de tráfego legítimo normal (`label == 0`) no treino sem vazamento de ataques ou parâmetros de contaminação.
- [x] Cálculo e persistência de $\tau_{\text{DIF, ref}}$ e hiperparâmetros em `models/dif_metadata.json`.
- [x] Script CLI `src/score_dif.py` com argumentos `--latent_parquet`, `--model_path`, `--output_parquet`, `--batch_size`, `--n_jobs`.
- [x] Persistência de `data/processed/dif_scores.parquet` com colunas `score_dif_deas`, `score_dif_standard`, rótulos originais e índices.
- [x] Teste de integração ponta a ponta executando treino e inferência a partir de um parquet sintético temporário.

# Framework Híbrido NIDS — FC-DAE + Deep Isolation Forest (Pre-IF)

Este repositório contém a implementação do framework híbrido proposto para Detecção de Intrusões em Redes (NIDS), combinando um **Fully Connected Deep Autoencoder (FC-DAE)** com a arquitetura **Deep Isolation Forest (DIF)** através do pipeline de integração **Pre-IF**.

---

## 📌 Status do Projeto e Ponto de Parada

> [!IMPORTANT]
> **Etapa Atual:** Conclusão completa de todos os 5 módulos do framework híbrido NIDS (Módulos 1 a 5). A infraestrutura de avaliação experimental, particionamento sem vazamento (*anti-leakage*), cadeia de baselines (B1 a B6b), ablação do DEAS, detector de fusão linear ponderada adaptativo, regra disjuntiva (OR) e análise de sensibilidade OFAT foram implementados e validados com 100% de aprovação na suíte de testes.

| Módulo | Descrição | Status | Artefatos Principais |
| :--- | :--- | :---: | :--- |
| **Módulo 1** | Pré-processamento e Normalização Min-Max $[0, 1]$ | `[x] Concluído` | `X_train_autoencoder.parquet` |
| **Módulo 2** | Arquitetura Deep Autoencoder (FC-DAE em PyTorch) | `[x] Concluído` | [ae.md](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/ae.md), [autoencoder.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/models/autoencoder.py), [train_ae.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/train_ae.py), [extract_latent.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/extract_latent.py) |
| **Módulo 3** | Pipeline de Transição Pre-IF e Validação Latente | `[x] Concluído` | [pre_if.md](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/pre_if.md), [validate_latent.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/validate_latent.py), [validation_report.md](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/reports/validation_report.md) |
| **Módulo 4** | Deep Isolation Forest (DIF) sobre Espaço Latente | `[x] Concluído` | [dif.md](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/dif.md), [dif.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/models/dif.py), [train_dif.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/train_dif.py), [score_dif.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/score_dif.py) |
| **Módulo 5** | Avaliação Experimental Completa e Métricas de Detecção | `[x] Concluído` | [eval.md](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/eval.md), [evaluate_protocol.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/evaluate_protocol.py), [partitioner.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/evaluation/partitioner.py), [baselines.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/evaluation/baselines.py), [hybrid.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/evaluation/hybrid.py), [sensitivity.py](file:///c:/Users/robso/Downloads/dea-dif-artefact-main/dea-dif-artefact-main/src/evaluation/sensitivity.py) |


---

## 📂 Estrutura do Repositório

```text
d:\tcc-II\
├── README.md                                   # Documentação geral do projeto
├── CONTEXT.md                                  # Glossário de termos e modelo de domínio
├── ae.md                                       # Especificação detalhada do Módulo 2 (FC-DAE)
├── pre_if.md                                   # Especificação detalhada do Módulo 3 (Pre-IF)
├── docs/
│   └── adr/
│       ├── 0001-fc-dae-linear-bottleneck-pytorch.md   # ADR 0001: Decisão da arquitetura FC-DAE
│       └── 0002-pre-if-latent-space-validation-and-scaling.md # ADR 0002: Validação do Espaço Latente
├── models/
│   └── fc_dae_best.pt                          # Pesos do FC-DAE treinado (MSE: 0.000301)
├── data/
│   └── processed/
│       └── latent_space.parquet                # Espaço latente (1.589.924 x 11) extraído
├── reports/
│   ├── validation_report.md                    # Relatório de validação do Módulo 3
│   └── figures/
│       ├── tsne_binary.png                     # Projeção t-SNE 2D (Visão Binária)
│       ├── tsne_multiclass.png                 # Projeção t-SNE 2D (Visão Multiclasse)
│       └── reconstruction_error_kde.png        # Curva de densidade KDE do Erro MSE
└── src/
    ├── data/
    │   ├── dataset.py                          # PyTorch TrafficParquetDataset
    │   └── X_train_autoencoder.parquet         # Dataset pré-processado CICIDS2017/NSL-KDD
    ├── models/
    │   └── autoencoder.py                      # Classe FCDAE(nn.Module)
    ├── train_ae.py                             # Script de treinamento do FC-DAE
    ├── extract_latent.py                       # Script de extração de z e cálculo de MSE
    └── validate_latent.py                      # Script de validação estatística e t-SNE
```

---

## 🛠️ Especificações Técnicas Implementadas

### Módulo 2 — Fully Connected Deep Autoencoder (FC-DAE)
* **Topologia Simétrica:** $n \rightarrow n/2 \rightarrow n/4 \rightarrow m \rightarrow n/4 \rightarrow n/2 \rightarrow n$ neurônios.
* **Ativações:** `ReLU` nas camadas ocultas, **Linear** no *bottleneck* (preservação de continuidade geométrica) e **Sigmoide** na saída.
* **Bottleneck Dinâmico:** $m = \lfloor n/8 \rfloor = 9$ (para $n=77$ atributos do dataset).
* **Treinamento:** Otimizador `Adam` ($lr=10^{-3}$), `MSELoss`, `ReduceLROnPlateau` e `EarlyStopping` (paciência = 10 épocas).
* **Desempenho de Treinamento:** Concluído em 50 épocas sobre $1.589.924$ amostras normais com menor perda de validação de **`0.000301`**.

### Módulo 3 — Integração Pre-IF e Validação Latente
* **Extração:** Geração da matriz de espaço latente $z \in \mathbb{R}^9$ acompanhada da coluna de erro MSE por amostra.
* **Validação Gráfica:** Projeção não-linear t-SNE (amostragem estratificada de 5.000 pontos) gerando visualizações binárias e multiclasse.
* **Reconstrução Diferencial:** Histograma/KDE sobreposto do Erro MSE, com suporte ao teste de Kolmogorov-Smirnov (KS) e cálculo da métrica ROC-AUC.

---

## 🚀 Como Executar o Projeto

### 1. Requisitos do Sistema
* Python 3.10+
* Pacotes necessários: `torch`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`

### 2. Treinamento do Autoencoder (Módulo 2)
```powershell
python src/train_ae.py --parquet_path src/data/X_train_autoencoder.parquet --epochs 50 --batch_size 2048
```

### 3. Extração do Espaço Latente
```powershell
python src/extract_latent.py --model_path models/fc_dae_best.pt --parquet_path src/data/X_train_autoencoder.parquet --output_parquet_path data/processed/latent_space.parquet
```

### 4. Validação do Módulo Pre-IF (Módulo 3)
```powershell
python src/validate_latent.py --latent_parquet data/processed/latent_space.parquet --mse_column mse --sample_size 5000
```

### 5. Treinamento do Deep Isolation Forest (Módulo 4)
```powershell
python src/train_dif.py --latent_parquet data/processed/latent_space.parquet --model_out models/dif_ensemble.joblib --metadata_out models/dif_metadata.json --trees 100 --subsample 256 --representations 10 --lambda_deas 0.5 --val_percentile 95.0
```

### 6. Inferência e Extração de Scores DIF (Módulo 4)
```powershell
python src/score_dif.py --latent_parquet data/processed/latent_space.parquet --model_path models/dif_ensemble.joblib --output_parquet data/processed/dif_scores.parquet --batch_size 32768 --n_jobs -1
```

### 7. Avaliação Experimental de Baselines e Pipeline Unificado (Módulo 5)
```powershell
# Execução padrão do pipeline de avaliação de baselines (B1 a B6b) com dados sintéticos
python src/evaluate_protocol.py --synthetic --output_dir reports/ --random_state 42 --run_ofat

# Execução sobre o dataset parquet processado do CICIDS2017
python src/evaluate_protocol.py --data_path data/processed/cicids2017_full.parquet --output_dir reports/ --random_state 42 --run_ofat --run_lstm
```

---

## 📋 Próximas Etapas e Conclusão Metodológica

1. **Geração de Gráficos Finais para o TCC:**
   - Consolidar as figuras exportadas em `reports/figures/sensitivity/` nos capítulos 4 e 5 do TCC.
   - Integrar as tabelas Markdown geradas (`reports/experimental_results.md`) diretamente na redação dos resultados experimentais.
2. **Execução em Larga Escala:**
   - Rodar o pipeline unificado com `--data_path` apontando para o dataset completo em ambiente de produção com múltiplos nós/GPUs.


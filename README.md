# Framework Híbrido NIDS — FC-DAE + Deep Isolation Forest (Pre-IF)

Este repositório contém a implementação do framework híbrido proposto para Detecção de Intrusões em Redes (NIDS), combinando um **Fully Connected Deep Autoencoder (FC-DAE)** com a arquitetura **Deep Isolation Forest (DIF)** através do pipeline de integração **Pre-IF**.

---

## 📌 Status do Projeto e Ponto de Parada

> [!IMPORTANT]
> **Etapa Atual de Parada:** Conclusão completa do **Módulo 3 (Integração Pre-IF e Validação do Espaço Latente)**. O modelo FC-DAE foi treinado com sucesso (perda MSE: `0.000301`), o espaço latente $z \in \mathbb{R}^9$ e o erro MSE foram extraídos para $1.589.924$ amostras e os relatórios/gráficos t-SNE foram gerados em `reports/`.
> 
> 📍 **Próximo Passo:** Implementação do **Módulo 4 (Deep Isolation Forest - DIF)** sobre o espaço latente persistido em `data/processed/latent_space.parquet`.

| Módulo | Descrição | Status | Artefatos Principais |
| :--- | :--- | :---: | :--- |
| **Módulo 1** | Pré-processamento e Normalização Min-Max $[0, 1]$ | `[x] Concluído` | [X_train_autoencoder.parquet](file:///d:/tcc-II/src/data/X_train_autoencoder.parquet) |
| **Módulo 2** | Arquitetura Deep Autoencoder (FC-DAE em PyTorch) | `[x] Concluído` | [ae.md](file:///d:/tcc-II/ae.md), [autoencoder.py](file:///d:/tcc-II/src/models/autoencoder.py), [train_ae.py](file:///d:/tcc-II/src/train_ae.py), [extract_latent.py](file:///d:/tcc-II/src/extract_latent.py) |
| **Módulo 3** | Pipeline de Transição Pre-IF e Validação Latente | `[x] Concluído` | [pre_if.md](file:///d:/tcc-II/pre_if.md), [validate_latent.py](file:///d:/tcc-II/src/validate_latent.py), [validation_report.md](file:///d:/tcc-II/reports/validation_report.md) |
| **Módulo 4** | Deep Isolation Forest (DIF) sobre Espaço Latente | `[ ] Pendente` | *(Próxima etapa)* |
| **Módulo 5** | Avaliação Experimental Completa e Métricas de Detecção | `[ ] Pendente` | *(Etapa final)* |

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

---

## 📋 Próximas Etapas

1. **Implementar Módulo 4 (Deep Isolation Forest - DIF):**
   - Construção do módulo PyTorch / Python para projetar aleatoriamente o espaço latente $z \in \mathbb{R}^9$ em subespaços e calcular a profundidade média de isolamento.
2. **Executar Pipeline Ponta a Ponta no Conjunto de Teste:**
   - Ingerir o dataset de teste contendo tráfego benigno e anomalias (`X_test.parquet`) para obter a matriz de confusão final, F1-Score, Precisão, Recall e curva ROC-AUC do framework completo.

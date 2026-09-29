# Framework Híbrido NIDS — FC-DAE + Deep Isolation Forest (Pre-IF)

Este repositório contém a implementação do framework híbrido proposto para Detecção de Intrusões em Redes (NIDS), combinando um **Fully Connected Deep Autoencoder (FC-DAE)** com a arquitetura **Deep Isolation Forest (DIF)** através do pipeline de integração **Pre-IF**.

---

## 📌 Status do Projeto e Ponto de Parada

> [!IMPORTANT]
> **Etapa Atual:** Conclusão completa de todos os 5 módulos do framework híbrido NIDS (Módulos 1 a 5). A infraestrutura de avaliação experimental, particionamento sem vazamento (*anti-leakage*), cadeia de baselines, ablação do DEAS, detector de fusão linear ponderada adaptativo, regra disjuntiva (OR) e análise de sensibilidade OFAT foram implementados e validados com 100% de aprovação na suíte de testes.

### Cadeia de Baselines — 7 Executados, 1 Bloqueado

| ID | Modelo | Entrada | DEAS | Executado |
| :---: | :--- | :--- | :---: | :---: |
| **B1** | iForest isolado | features brutas $x \in \mathbb{R}^n$ | — | `[x]` |
| **B2** | FC-DAE isolado (erro MSE) | features brutas | — | `[x]` |
| **B3** | Pre-IF + iForest clássico | latente $z \in \mathbb{R}^m$ | não | `[x]` |
| **B3b** | Pre-IF + DIF puro (ablação) | latente | $\lambda = 0$ | `[x]` |
| **B4** | Random Forest supervisionado | features brutas | — | `[x]` |
| **B5** | LSTM-AE recorrente | janelas temporais | — | `[ ]` **bloqueado** |
| **B6a** | Fusão linear ponderada ($\alpha$ calibrado) | latente + MSE | $\lambda > 0$ | `[x]` |
| **B6b** | Regra disjuntiva (OR) | latente + MSE | $\lambda > 0$ | `[x]` |

> [!WARNING]
> **B5 não é executável com os artefatos atuais.** O janelamento cronológico ($W = 10$, passo $S = 1$, rótulo em $t_W$) exige a coluna `Timestamp`. Nenhuma das features dos parquets processados é temporal e **não existe** arquivo paralelo de timestamps/metadados em nenhum dos três diretórios de dataset — `eval.md` §4.8.5 prevê `timestamps.parquet`, e o item correspondente do checklist §4.8.7 segue desmarcado. Destravar B5 exige reexecutar o Módulo 1 preservando a coluna temporal original. A rotina de treino detecta essa ausência e registra a ressalva em cada relatório.

### Auditoria do DEAS — o que a evidência sustenta

A varredura OFAT de $\lambda$ sai quase plana, o que levanta a suspeita legítima de que
$\lambda$ não estaria chegando ao cálculo do escore. **Não é o caso — a fiação está
correta**, e foi verificada em quatro frentes:

| Verificação | Resultado |
| :--- | :--- |
| $\lambda$ do construtor chega ao escore | sim (média 0,5075 → 0,6073 para $\lambda$ 0 → 1) |
| `score_samples(lambda_deas=...)` sobrepõe o construtor | sim |
| $\lambda = 0$ reproduz `score_dif_standard` | exato (diferença máxima 0,0) |
| A floresta independe de $\lambda$ (só o escore muda) | sim, confirmado |

O que a auditoria revelou é outra coisa, e mais séria para o texto:

> [!CAUTION]
> **`score_deas >= score_standard` é uma identidade algébrica, não um resultado.**
> Como $h_{\text{deas}} = \sum(1 - \lambda \cdot \text{dev}) + c_{\text{folha}}$ com
> $\text{dev} \ge 0$, sempre vale $h_{\text{deas}} \le h_{\text{standard}}$; e o escore
> $2^{-h/c}$ é decrescente em $h$. Logo o DEAS eleva o escore de **todo** ponto — denso,
> vazio ou extremo. Qualquer teste ou argumento da forma "na ghost region o DEAS acusa
> mais anomalia que o padrão" é verdadeiro por construção e **não** evidencia mitigação.

A afirmação não-trivial — e que de fato se sustenta no cenário sintético de dois
clusters — é que o DEAS eleva a região vazia **mais** do que a densa
(+0,0830 contra +0,0372), aumentando a separação de 0,2425 para 0,2883. É essa a forma
correta de enunciar a hipótese, e é o que a suíte passou a testar.

**Nos dados reais, porém, o efeito líquido em AUC é levemente negativo.** Em 5 de 6
combinações dataset × semente o AUC cai ao ir de $\lambda = 0$ para $\lambda = 1$
(entre −0,0014 e −0,0092); a exceção é +0,0006, dentro do ruído. A explicação é
estrutural: a redução total de $h$ é $\text{dev} \times \text{profundidade}$, e as duas
grandezas se opõem — anomalias têm desvio por nó maior (0,62 contra 0,42) mas caminhos
mais curtos (5,6 contra 7,8 nós). Os produtos quase se anulam (3,216 contra 3,239),
marginalmente a favor dos normais, de modo que o DEAS funciona como deslocamento quase
uniforme e comprime de leve a separação.

Consequência para o TCC: o DEAS, como formulado, **não** melhora o ranqueamento nestas
três bases. Isso é reportável como resultado negativo honesto — e é coerente com o
$\Delta_{\text{DEAS}}$ negativo do UNSW-NB15.

> [!NOTE]
> A contagem "B1 a B6b" corresponde a **8 identificadores** (`B1, B2, B3, B3b, B4, B5, B6a, B6b`), dos quais 7 são executados. `eval.md` §4.8.2 já lista os oito; se o texto do TCC na Seção 4.8.2 ainda enumerar seis sem `B3b`/`B6a`/`B6b`, é o texto que precisa ser sincronizado. Atenção a uma divergência de rotulagem: em `eval.md` §4.8.2, **B6a** designa "DIF com DEAS" e **B6b** o "framework híbrido completo"; no código, **B6a** é a fusão linear ponderada e **B6b** a regra OR (ambos já híbridos). Uma das duas convenções deve prevalecer.

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
dea-dif-artefact/
├── README.md                                   # Documentação geral do projeto
├── CONTEXT.md                                  # Glossário de termos e modelo de domínio
├── ae.md / pre_if.md / dif.md / eval.md        # Especificações dos Módulos 2 a 5
├── requirements.txt                            # Dependências (pip install -r)
├── pytest.ini                                  # Coloca a raiz no sys.path para os testes
├── docs/adr/                                   # Architecture Decision Records
├── models/<DATASET>/fc_dae_best.pt             # Checkpoints por dataset
├── reports/                                    # Relatórios gerados pela rotina de treino
└── src/
    ├── train_pipeline.py                       # ★ Rotina completa de treino e validação (Módulos 2 a 5)
    ├── make_tcc_assets.py                      # ★ Figuras e tabelas prontas para os Capítulos 4 e 5
    ├── data/
    │   ├── dataset.py                          # PyTorch TrafficParquetDataset (parquet único)
    │   └── nids_datasets.py                    # ★ Carregamento e particionamento leakage-free multi-dataset
    ├── training/
    │   ├── ae_trainer.py                       # ★ Loop de treino/validação do FC-DAE
    │   ├── latent.py                           # ★ Extração de z e do MSE por fluxo
    │   ├── calibration.py                      # ★ Limiares tau_AE e tau_DIF_ref
    │   └── device.py                           # ★ Resolução CUDA / DirectML / CPU
    ├── models/
    │   ├── autoencoder.py                      # Classe FCDAE(nn.Module)
    │   └── dif.py                              # Deep Isolation Forest + escore DEAS
    ├── evaluation/
    │   ├── partitioner.py                      # Particionador de parquet monolítico
    │   ├── evaluator.py                        # Métricas da Tabela 4 e recall estratificado
    │   ├── baselines.py                        # Cadeia de baselines B1 a B6b
    │   ├── hybrid.py                           # Fusão linear ponderada e regra OR
    │   ├── sensitivity.py                      # Análise de sensibilidade OFAT
    │   ├── reporting.py                        # ★ Relatórios Markdown por dataset e consolidado
    │   ├── figures.py                          # ★ Figuras de publicação (paleta validada)
    │   └── tables.py                           # ★ Tabelas em LaTeX e Markdown
    ├── baselines/lstm_ae.py                    # Baseline sequencial B5 (LSTM-AE)
    ├── train_ae.py / extract_latent.py         # Scripts individuais do Módulo 2/3
    ├── validate_latent.py                      # Validação estatística e t-SNE
    ├── train_dif.py / score_dif.py             # Scripts individuais do Módulo 4
    └── evaluate_protocol.py                    # Avaliação sobre parquet monolítico (Módulo 5)
```

`★` = módulos da rotina de treino e validação multi-dataset.

---

## 🛠️ Especificações Técnicas Implementadas

### Módulo 2 — Fully Connected Deep Autoencoder (FC-DAE)
* **Topologia Simétrica:** $n \rightarrow n/2 \rightarrow n/4 \rightarrow m \rightarrow n/4 \rightarrow n/2 \rightarrow n$ neurônios.
* **Ativações:** `ReLU` nas camadas ocultas, **Linear** no *bottleneck* (preservação de continuidade geométrica) e **Sigmoide** na saída.
* **Bottleneck Dinâmico:** $m = \lfloor n/4 \rfloor$ para $n < 50$ e $m = \lfloor n/8 \rfloor$ para $n \ge 50$. Como o projeto é multi-dataset, $m$ varia por base:

  | Dataset | $n$ | $m$ | Treino benigno (`X_train_autoencoder`) |
  | :--- | ---: | ---: | ---: |
  | CICIDS2017 | 77 | **9** | 1.327.670 |
  | NSL-KDD | 115 | **14** | 53.876 |
  | UNSW-NB15 | 194 | **24** | 60.005 |

* **Treinamento:** Otimizador `Adam` ($lr=10^{-3}$), `MSELoss`, `ReduceLROnPlateau` e `EarlyStopping` (paciência = 10 épocas), exclusivamente sobre tráfego benigno.
* **Limiar de Anomalia:** $\tau_{AE}$ é calibrado no MSE do tráfego benigno de validação pelo **percentil $p$, com $p = 95$** por padrão (`--ae_threshold_rule percentile`), coerente com a Eq. 4 / Seção 4.7.1 e com a varredura OFAT do fator $p \in \{90, 95, 97, 99\}$. A formulação alternativa $\mu + k\sigma$ registrada em `ae.md` permanece disponível (`--ae_threshold_rule mu_sigma`) e é sempre reportada em paralelo.

> [!IMPORTANT]
> $\mu + 3\sigma$ **não** equivale ao percentil 99: a equivalência exigiria normalidade do erro de reconstrução, que é assimétrico à direita e de cauda longa. Medido nos artefatos, $\mu + 3\sigma$ cai em percentis empíricos **diferentes em cada base** — 95,73 (NSL-KDD) e 98,50 (UNSW-NB15) em execuções integrais, ≈97,1 (CICIDS2017) em execução subamostrada. Nenhum deles é 99, e o FPR resultante varia de 4,27% a 1,50%. Ou seja: $\mu+k\sigma$ é um ponto de operação *dependente do dataset*, enquanto o percentil $p$ fixa o FPR esperado em $100-p$ por construção. Cada relatório imprime as duas formulações lado a lado com o FPR esperado. `ae.md` (linha 41) e `CONTEXT.md` (linha 12) ainda descrevem "$\mu+3\sigma$ ou percentil 99" — redação a alinhar ao percentil $p=95$.

### Módulo 3 — Integração Pre-IF e Validação Latente
* **Extração:** Geração da matriz de espaço latente $z \in \mathbb{R}^m$ acompanhada da coluna de erro MSE por amostra.
* **Validação Gráfica:** Projeção não-linear t-SNE (amostragem estratificada de 5.000 pontos) gerando visualizações binárias e multiclasse.
* **Reconstrução Diferencial:** Histograma/KDE sobreposto do Erro MSE, com suporte ao teste de Kolmogorov-Smirnov (KS) e cálculo da métrica ROC-AUC.

---

## 📥 Formato Esperado dos Datasets Processados

A rotina de treino consome um diretório por dataset, no layout produzido pelo Módulo 1:

```text
<raiz-dos-dados>/
├── NSL-KDD-processados/
│   ├── X_train_autoencoder.parquet   # tráfego estritamente benigno (subconjunto benigno do treino)
│   ├── X_train_baseline.parquet      # treino completo (benigno + ataques)
│   ├── y_train_baseline.parquet      # rótulos multiclasse do treino
│   ├── X_test.parquet                # conjunto de teste cego
│   └── y_test.parquet                # rótulos multiclasse do teste
├── UNSW-NB15-processados/            # (+ preprocessor.joblib, opcional)
└── CICIDS2017-processados/
```

A coluna de rótulo e o valor benigno são **detectados automaticamente** por dataset:

| Dataset | n features | Coluna de rótulo | Rótulo benigno | Famílias de ataque |
| :--- | ---: | :--- | :--- | ---: |
| NSL-KDD | 115 | `xAttack` | `Normal` | 4 |
| UNSW-NB15 | 194 | `attack_cat` | `Normal` | 9 |
| CICIDS2017 | 77 | `Label` | `BENIGN` | 14 |

Rótulos do CICIDS2017 que chegam com o caractere de substituição Unicode
(`"Web Attack � XSS"`) são normalizados para `"Web Attack - XSS"`.

`preprocessor.joblib` está presente nos três diretórios e é a fonte da verdade da
transformação do Módulo 1, mas **não é requerido pela rotina de treino** — os
parquets já chegam transformados. A rotina apenas registra sua presença nos metadados.

### Nenhum artefato temporal: B5 sem base

Os diretórios contêm exatamente **6 arquivos cada** (os 5 parquets acima mais
`preprocessor.joblib`). Não há `timestamps.parquet` nem `metadados_*.parquet`, e
nenhuma das features é temporal — verificado por varredura de nomes de arquivo e de
coluna nas três bases. O baseline B5 permanece bloqueado (ver seção de baselines).

### Procedência dos splits: os artefatos são refatiamentos, não os splits canônicos

> [!WARNING]
> Os três artefatos usam **refatiamento aleatório estratificado 70/30**, não os splits
> canônicos das bases. Nos três casos o teste é exatamente $\lfloor 0{,}30 \cdot N \rfloor$
> e as proporções de classe do treino se replicam no teste na quarta casa decimal, sem
> nenhuma família exclusiva do teste. **Consequência metodológica: o cenário *zero-day*
> não está sendo avaliado.**

| Dataset | Treino | Teste | Teste/Total | Máx. divergência de proporção | Famílias só no teste |
| :--- | ---: | ---: | ---: | ---: | :---: |
| NSL-KDD | 103.504 | 44.360 | 30,00% | 0,0009 p.p. | nenhuma |
| UNSW-NB15 | 110.938 | 47.545 | 30,00% | < 0,01 p.p. | nenhuma |
| CICIDS2017 | 1.563.214 | 669.950 | 30,00% | < 0,01 p.p. | nenhuma |

No NSL-KDD isso significa que o **`KDDTest+` canônico não está preservado**: a união
treino+teste soma 147.864 registros (contra 148.517 de `KDDTrain+` + `KDDTest+`, ou seja
após deduplicação), e o `KDDTest+` responderia por 15,18% do total — não 30%. O
`KDDTest+` desloca a distribuição deliberadamente (Normal ~43%, R2L ~12%) e introduz
17 tipos de ataque ausentes do treino; aqui a distribuição é idêntica à do treino.
Para avaliar generalização a ataques inéditos é preciso reconstruir o split canônico no
Módulo 1. A rotina de treino diagnostica a procedência automaticamente
(`split_provenance` nos metadados) e emite a ressalva em cada relatório.

---

## 🚀 Como Executar a Rotina de Treino e a Validação Experimental

> [!IMPORTANT]
> Todos os comandos desta seção são executados **da raiz do repositório** (o diretório
> que contém `README.md` e a pasta `src/`). O pipeline é invocado como módulo
> (`python -m src.train_pipeline`), não pelo caminho do arquivo.

### Passo 1 — Preparar o ambiente

```powershell
cd <raiz-do-repositorio>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/macOS, troque a ativação por `source .venv/bin/activate`.

Requisitos: **Python 3.10+**. A execução em CPU é suficiente para as três bases
(nenhuma GPU é necessária); `--device` aceita `auto`, `cpu`, `cuda`, `cuda:0` ou `dml`
(GPU AMD/Intel no Windows, exige o pacote opcional `torch-directml`).

Confirme a instalação rodando a suíte de testes — ela não depende dos datasets:

```powershell
pytest -q
```

Saída esperada: `73 passed`. Um aviso `PytestUnhandledThreadExceptionWarning` na
captura de saída de subprocesso é conhecido e inofensivo.

### Passo 2 — Apontar para os datasets

Os parquets processados **não** ficam no repositório. Guarde os três diretórios sob uma
raiz única (o layout exigido está em *Formato Esperado dos Datasets Processados*):

```text
C:\dados\processados\
├── NSL-KDD-processados\
├── UNSW-NB15-processados\
└── CICIDS2017-processados\
```

O pipeline descobre sozinho todo diretório que contenha os 4 parquets obrigatórios
(`X_train_baseline`, `y_train_baseline`, `X_test`, `y_test`), detecta a coluna de rótulo
e o valor benigno de cada base, e aborta com mensagem explícita se algum arquivo faltar.

### Passo 3 — Ensaio rápido (menos de 2 minutos)

Antes da execução oficial, valide a cadeia inteira com poucas épocas e subamostragem.
Serve para conferir caminhos, memória e permissões de escrita sem esperar o treino real:

```powershell
python -m src.train_pipeline --data_root "C:\dados\processados" --epochs 5 --max_train_rows 50000 --max_test_rows 20000 --output_dir reports_smoke --models_dir models_smoke
```

Se os três datasets aparecerem em `Datasets concluídos: 3 | falhas: 0`, o ambiente está
pronto. Descarte `reports_smoke/` e `models_smoke/`: os relatórios trazem a ressalva de
subamostragem e os números **não** são comparáveis a uma execução integral.

### Passo 4 — Rotina de treino oficial

Esta é a execução que produz os resultados do TCC. A semente padrão (`--random_state 42`)
torna o particionamento, os pesos iniciais e o ensemble reprodutíveis.

```powershell
# Todas as bases encontradas sob a raiz de dados
python -m src.train_pipeline --data_root "C:\dados\processados" --epochs 60 --patience 10 --output_dir reports --models_dir models
```

```powershell
# Apenas as bases menores (NSL-KDD e UNSW-NB15), se quiser rodar o CICIDS2017 à parte
python -m src.train_pipeline --data_root "C:\dados\processados" --datasets "NSL-KDD,UNSW-NB15" --epochs 60 --patience 10
```

```powershell
# CICIDS2017 isolado: 1,33 milhão de fluxos benignos, batch maior para reduzir o número de passos
python -m src.train_pipeline --dataset_dir "C:\dados\processados\CICIDS2017-processados" --epochs 40 --patience 8 --batch_size 4096
```

Um dataset que falhe é registrado em `failures` e **não** interrompe os demais.

**Custo observado** (CPU de 12 threads, sem GPU, execução integral):

| Dataset | Treino benigno | Tempo total | Pico de memória |
| :--- | ---: | ---: | ---: |
| NSL-KDD | 45.795 | ~45 s | < 1 GB |
| UNSW-NB15 | 51.004 | ~52 s | < 1 GB |
| CICIDS2017 | 1.128.520 | dezenas de minutos | ~1,8 GB |

O que a rotina faz, em ordem, para cada dataset:

| Etapa | Módulo | O que faz |
| :--- | :--- | :--- |
| 1 | — | Particionamento *leakage-free*: o teste do artefato permanece cego; a validação sai de `X_train_baseline` |
| 2 | 2 | Treino do FC-DAE no tráfego benigno, validado em partição benigna disjunta (early stopping + `ReduceLROnPlateau`, melhor época restaurada) |
| 3 | 3 | Extração de `z` e do MSE por fluxo; calibração de `tau_AE` = percentil `p=95` do MSE benigno |
| 4 | 4 | Indução do DIF sobre latentes benignos; calibração de `tau_DIF_ref` por percentil |
| 5 | 5 | Baselines no teste cego, recall estratificado, ablação do DEAS e OFAT opcional |

### Passo 5 — Validação experimental (Módulo 5)

A avaliação dos baselines no teste cego **já ocorre** na etapa 5 do Passo 4: não há
comando separado para a Tabela 4. O que se acrescenta aqui são as análises opcionais.

**Sensibilidade OFAT** sobre os fatores que não exigem retreinar o autoencoder
(`L`, `t`, `p`, `lambda`), com as figuras por fator:

```powershell
python -m src.train_pipeline --data_root "C:\dados\processados" --epochs 60 --run_ofat
```

**Sensibilidade à dimensão latente `m`.** Cada valor de `m` muda a própria
representação, então exige um FC-DAE novo, nova extração latente e novo DIF — por isso
é opt-in e bem mais caro:

```powershell
python -m src.train_pipeline --dataset_dir "C:\dados\processados\CICIDS2017-processados" --ofat_m_values 9 12 19 38 --ofat_m_epochs 15
```

**Espaço latente persistido** para as figuras t-SNE e a análise de reconstrução
diferencial do Módulo 3:

```powershell
python -m src.train_pipeline --data_root "C:\dados\processados" --epochs 60 --save_latent
python src/validate_latent.py --latent_parquet reports/NSL-KDD/latent_space_test.parquet --mse_column mse --sample_size 5000
```

### Passo 6 — Gerar as figuras e tabelas do TCC

`src/make_tcc_assets.py` lê o que a rotina já gravou em `reports/` — **sem retreinar
nada** — e consolida um pacote pronto para os Capítulos 4 e 5:

```powershell
python -m src.make_tcc_assets --reports_dir reports
```

Para o pacote completo, a rotina do Passo 4 precisa ter gravado os artefatos opcionais:

```powershell
python -m src.train_pipeline --data_root "C:\dados\processados" --epochs 60 --run_ofat --save_scores --save_latent
python -m src.make_tcc_assets --reports_dir reports
```

| Figura | Exige | Capítulo |
| :--- | :--- | :--- |
| Convergência do FC-DAE | — | 4 |
| Reconstrução diferencial (MSE benigno vs. ataque, com `tau_AE`) | `--save_scores` ou `--save_latent` | 4 |
| Sensibilidade OFAT (um gráfico por fator) | `--run_ofat` | 4 |
| Curvas ROC da cadeia de ablação | `--save_scores` | 5 |
| Mapa de calor do recall por família | — | 5 |
| Ablação do DEAS (λ=0 vs. DEAS) | ≥ 1 dataset | 5 |
| Comparativo entre datasets (F1 e AUC) | ≥ 2 datasets | 5 |

Artefatos ausentes não quebram a geração: a figura é omitida e o motivo fica
registrado em `MANIFEST.md`.

O gerador também **reconstrói `reports/consolidated_results.md`** a partir de todos os
JSONs encontrados. Isso importa quando as bases são rodadas em invocações separadas: a
rotina de treino grava o consolidado apenas com os datasets daquela execução, e o
arquivo acaba incompleto. Use `--no_refresh_consolidated` para desativar; com
`--datasets` a reconstrução é pulada automaticamente, para não truncar o arquivo.

> [!CAUTION]
> O gerador detecta **fusão degenerada**: quando o `alpha` calibrado cai em 0 ou 1, B6a
> deixa de ser híbrido e colapsa no autoencoder isolado (B2) ou no DIF puro, e o
> `Delta_DEAS` **deixa de medir a contribuição do DEAS**. O caso é marcado no relatório
> do dataset, na tabela consolidada, na própria figura de ablação e no `MANIFEST.md`.
> Nos artefatos atuais isso ocorre no **CICIDS2017** (`alpha = 0`).

```text
reports/tcc_assets/
├── MANIFEST.md              # mapa arquivo -> capítulo/seção do TCC
├── figuras/*.png            # 300 dpi, modo claro
└── tabelas/*.tex, *.md      # cada tabela nos dois formatos
```

Cada tabela sai em **`.tex`** (para `\input{}` em LaTeX) e **`.md`** (para colar em
Word/Google Docs), com números idênticos:

```latex
\input{tabelas/tabela4_NSL-KDD.tex}
\includegraphics[width=\linewidth]{figuras/curvas_roc_NSL-KDD.png}
```

> [!NOTE]
> As figuras identificam cada série por **cor e** marcador/hachura, e cada célula do
> mapa de calor traz o valor impresso, de modo que nada depende só da cor — requisito
> para impressão monocromática e para leitores com deficiência de visão de cores. A
> paleta é validada por `validate_palette.js` nas combinações usadas (3 slots em
> `--pairs all` para as curvas ROC sobrepostas, 4 no pairlist adjacente para o OFAT).

> [!WARNING]
> O gerador **recusa** relatórios de versões anteriores da rotina (por exemplo, de
> antes da mudança de `tau_AE` para o percentil `p=95`), porque consolidá-los produziria
> tabelas em que datasets diferentes foram calibrados por regras diferentes. Reexecute a
> rotina nesses datasets, ou use `--allow_stale` cientes de que deixam de ser
> comparáveis. Execuções subamostradas geram um aviso e não devem ir para o TCC como
> resultado final.

### Passo 7 — Ler os resultados

```text
reports/
├── consolidated_results.md            # comparação entre as três bases (comece por aqui)
├── consolidated_results.json
└── <DATASET>/
    ├── experimental_results.md        # relatório completo da base
    ├── experimental_results.json      # os mesmos dados, para gráficos e tabelas
    ├── fc_dae_training_history.json   # perda de treino/validação e lr por época
    ├── latent_space_{val,test}.parquet# apenas com --save_latent
    ├── test_scores.parquet            # apenas com --save_scores (insumo das curvas ROC)
    ├── ofat_latent_dim_sweep.json     # apenas com --ofat_m_values
    └── figures/sensitivity/*.png      # apenas com --run_ofat
reports/tcc_assets/                    # gerado pelo Passo 6
models/<DATASET>/fc_dae_best.pt        # checkpoint da melhor época (inclui feature_columns)
```

Seções de `experimental_results.md`, na ordem em que aparecem:

| Seção | Conteúdo | Uso no TCC |
| :--- | :--- | :--- |
| 1 e 1.1 | Partições, particionamento efetivo e procedência do split | Seção 4.8.1 |
| 2 | Treino do FC-DAE e as duas formulações de `tau_AE` | Seção 4.7 |
| 3 | Hiperparâmetros do DIF e `tau_DIF_ref` | Seção 4.7.2 |
| 4 | **Tabela 4** — métricas comparativas no teste cego | Tabela 4 |
| 5 | Ablação do DEAS (`Delta_DEAS` em F1 e AUC) | Seção 4.8.2 |
| 6 | Taxa de detecção estratificada por família, com ⚠️ nas classes raras | Seção 4.8.3 |
| 7 | Limiares e `alpha` calibrados, separados por regime | Seção 4.8.4 |
| 8 | Varreduras OFAT | Seção 4.8.6 |

### Passo 8 — Parâmetros de ajuste

`python -m src.train_pipeline --help` lista todos. Os principais:

| Grupo | Parâmetros (padrão) |
| :--- | :--- |
| Protocolo | `--val_normal_ratio` (0.15), `--val_attack_ratio` (0.20), `--rare_threshold` (15), `--random_state` (42) |
| FC-DAE | `--epochs` (50), `--batch_size` (2048), `--lr` (1e-3), `--patience` (10), `--bottleneck_dim` (auto), `--device` (auto) |
| $\tau_{AE}$ | `--ae_threshold_rule` (`percentile`), `--ae_percentile` (95), `--k_sigma` (3.0) |
| DIF | `--trees` (100), `--subsample` (256), `--representations` (10), `--n_layers` (3), `--projection_dim` (= m), `--lambda_deas` (0.5), `--val_percentile` (95) |
| Avaliação | `--alpha_grid` (101), `--p_low`/`--p_high` (1/99), `--max_supervised_rows` (300000), `--run_ofat`, `--ofat_m_values` |
| Escala | `--max_train_rows`, `--max_test_rows`, `--score_batch_size` (32768), `--inference_batch_size` (16384), `--n_jobs` (-1) |

### Solução de problemas

| Sintoma | Causa e correção |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'src'` | Comando executado de outro diretório. Vá para a raiz do repositório e use `python -m src.train_pipeline`. |
| `Informe --data_root ou --dataset_dir` | Nenhuma fonte de dados indicada. Passe uma das duas opções. |
| `Nenhum dataset válido encontrado sob ...` | A raiz não contém diretórios com os 4 parquets obrigatórios. Confira o layout do Passo 2. |
| `Nenhum rótulo benigno reconhecido entre [...]` | Rótulo benigno fora da lista conhecida. Informe `--normal_label "<valor>"` (e `--label_column` se preciso). |
| `MemoryError` ou travamento no CICIDS2017 | Reduza a escala: `--max_train_rows 400000 --max_test_rows 200000`, e baixe `--max_supervised_rows` (o B4 é o baseline mais custoso). |
| Treino termina em `Melhor época = <última>` | O early stopping não acionou: o modelo ainda estava melhorando. Aumente `--epochs`. |
| `Nenhum ataque disponível na validação` | Base sem ataques no treino, ou `--val_attack_ratio` muito baixo para famílias minúsculas. |

### Scripts Individuais por Módulo

A rotina do Passo 4 substitui os scripts abaixo, mantidos para depuração de um módulo
isolado. Eles operam sobre **um** parquet e não implementam o protocolo *leakage-free*.

```powershell
# Módulo 2 — treino do FC-DAE direto no parquet benigno (sem partição de validação rotulada)
python src/train_ae.py --parquet_path "C:\dados\processados\NSL-KDD-processados\X_train_autoencoder.parquet" --output_model_path models/fc_dae_debug.pt --epochs 50 --batch_size 2048

# Módulo 3 — extração do espaço latente e do MSE
python src/extract_latent.py --model_path models/fc_dae_debug.pt --parquet_path "C:\dados\processados\NSL-KDD-processados\X_train_autoencoder.parquet" --output_parquet_path data/processed/latent_space.parquet

# Módulo 3 — validação estatística e t-SNE
python src/validate_latent.py --latent_parquet data/processed/latent_space.parquet --mse_column mse --sample_size 5000

# Módulo 4 — treino do Deep Isolation Forest
python src/train_dif.py --latent_parquet data/processed/latent_space.parquet --model_out models/dif_ensemble.joblib --metadata_out models/dif_metadata.json --trees 100 --subsample 256 --representations 10 --lambda_deas 0.5 --val_percentile 95.0

# Módulo 4 — inferência e extração de scores
python src/score_dif.py --latent_parquet data/processed/latent_space.parquet --model_path models/dif_ensemble.joblib --output_parquet data/processed/dif_scores.parquet --batch_size 32768 --n_jobs -1

# Módulo 5 — mecânica do pipeline sobre dados sintéticos
python src/evaluate_protocol.py --synthetic --output_dir reports_synthetic/ --random_state 42 --run_ofat
```

> [!WARNING]
> `src/evaluate_protocol.py --data_path <parquet>` **não** treina o FC-DAE: o "latente"
> é um recorte das `m` primeiras features brutas e o MSE é preenchido com zeros, o que
> invalida B2, B6a e B6b. O próprio script emite esse aviso. Para resultados válidos use
> `python -m src.train_pipeline`.

---

## 📋 Próximas Etapas e Conclusão Metodológica

1. **Geração de Gráficos Finais para o TCC:**
   - Consolidar as figuras exportadas em `reports/<DATASET>/figures/sensitivity/` nos capítulos 4 e 5 do TCC.
   - Integrar as tabelas Markdown geradas (`reports/<DATASET>/experimental_results.md` e `reports/consolidated_results.md`) diretamente na redação dos resultados experimentais.
2. **Execução Integral do CICIDS2017:**
   - Rodar o Passo 4 sem `--max_train_rows`/`--max_test_rows` na base completa (1,33 milhão de fluxos benignos de treino), preferencialmente com `--device cuda`.
3. **Sincronização do Texto do TCC** (divergências já mapeadas neste README):
   - Alinhar a formulação de $\tau_{AE}$ (percentil $p=95$) em `ae.md` e `CONTEXT.md`.
   - Sincronizar a Seção 4.8.2 com os 8 identificadores de baseline e resolver a rotulagem de `B6a`/`B6b`.
   - Registrar que os splits fornecidos são refatiamentos 70/30 e que o cenário *zero-day* não está sendo avaliado.
4. **Destravar o Baseline B5:**
   - Reexecutar o Módulo 1 preservando a coluna `Timestamp` para viabilizar o janelamento cronológico do LSTM-AE.


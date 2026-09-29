**Ablação do DEAS**

| Dataset | F1 B3b (%) | F1 B6a (%) | Δ F1 | AUC B3b | AUC B6a | Δ AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CICIDS2017 | 47.77 | 69.60 | +0.2183 | 0.8533 | 0.9036 | +0.0503 |
| NSL-KDD | 86.20 | 92.41 | +0.0621 | 0.9021 | 0.9703 | +0.0682 |
| UNSW-NB15 | 71.64 | 69.08 | -0.0256 | 0.7737 | 0.7712 | -0.0025 |

_B3b e B6a compartilham a mesma floresta de isolamento (a indução das árvores não depende de λ), de modo que o delta é atribuível exclusivamente ao DEAS._

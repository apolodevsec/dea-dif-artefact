**Particionamento efetivo**

| Dataset | n | m | Treino benigno | Validação | Teste | Benigno tr/val/te (%) | Ataque val/te (%) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CICIDS2017 | 77 | 9 | 1.128.520 | 246.259 | 669.950 | 59.5/10.5/30.0 | 14.0/30.0 |
| NSL-KDD | 115 | 14 | 45.795 | 18.007 | 44.360 | 59.5/10.5/30.0 | 14.0/30.0 |
| UNSW-NB15 | 194 | 24 | 51.004 | 19.188 | 47.545 | 59.5/10.5/30.0 | 14.0/30.0 |

_A fronteira treino/teste é fixada pelos artefatos processados, de modo que as frações efetivas diferem do esquema nominal 70/15/15; o teste permanece cego._

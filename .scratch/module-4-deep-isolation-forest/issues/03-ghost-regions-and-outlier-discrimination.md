# 03: Mitigação de Ghost Regions e Discriminação Geométrica

**What to build:** Suíte de testes e validação comportamental da hipótese central do TCC: mitigação de *ghost regions* pelo mecanismo DEAS. Em um cenário bimodal sintético com clusters densos separados por espaço vazio, pontos na *ghost region* recebem pontuação anômala significativamente superior via DEAS comparado ao iForest clássico, enquanto os pontos centrais permanecem com baixo escore de anomalia.

**Blocked by:** 02-deas-mechanism-and-analytical-ablation

**Status:** closed

- [x] Gerador de dados sintéticos bimodais simulando dois clusters densos e uma região intermediária de baixa densidade (*ghost region*).
- [x] Teste de atenuação de *ghost regions*: comprovação de que o score DEAS para amostras na região intermediária vazia é estatisticamente superior ao score do iForest padrão.
- [x] Teste de discriminação geométrica de outliers: garantia de que amostras afastadas dos centróides recebem escores anômalos substancialmente superiores a amostras no interior dos clusters normais.
- [x] Teste de preservação de normalidade: garantia de que amostras no núcleo dos clusters legítimos não sofrem falsos alarmes espúrios.

# 2. Estratégia de Validação e Escalonamento do Espaço Latente no Módulo Pre-IF

Data: 2026-09-09

## Status

Aceito

## Contexto

O Módulo 3 (Integração Pre-IF) atua como ponte entre a representação comprimida do FC-DAE ($z \in \mathbb{R}^m$) e o Deep Isolation Forest (Módulo 4). Faz-se necessário validar a hipótese de que o espaço latente preserva a separação geométrica entre tráfego benigno e anômalo, além de definir o tratamento de escalonamento ideal para ingestão posterior no DIF.

## Decisão

1. **Amostragem para t-SNE:** Adotar amostragem aleatória estratificada de $N = 5.000$ a $10.000$ pontos para projeção t-SNE 2D, contornando a complexidade $\mathcal{O}(N^2)$ sem perda de representatividade.
2. **Granularidade da Visualização:** Gerar projeções t-SNE sob duas visões: Binária (Benigno vs Anômalo) e Multiclasse (discriminando os tipos de ataque do dataset CICIDS2017/NSL-KDD).
3. **Métricas de Reconstrução Diferencial:** Quantificar a sensibilidade do Autoencoder combinando gráficos de densidade de probabilidade (KDE), estatística de teste de Kolmogorov-Smirnov (KS) e pontuação de ROC-AUC do erro de reconstrução MSE.
4. **Preservação de Escala ($z$ Bruto):** Manter a amplitude original do vetor $z$ gerado pelo *bottleneck* linear do Encoder, sem forçar padronização Z-Score, preservando a topologia contínua natural aprendida para o Deep Isolation Forest.
5. **Automação via Script:** Implementar o script `src/validate_latent.py` para gerar e persistir automaticamente todos os gráficos e relatórios numéricos em `reports/figures/` e `reports/`.

## Consequências

### Positivas
- Validação visual e quantitativa rigorosa das propriedades do espaço latente antes de alimentar o isolamento.
- Desempenho eficiente de geração de gráficos t-SNE sem travamentos de memória.
- Garantia de que a topologia não-linear destilada pelo Autoencoder é preservada intacta para as projeções do DIF.

### Negativas
- A amostragem de 5k-10k pontos para o t-SNE descarta uma fração do volume total de inferência apenas para fins de visualização gráfica.

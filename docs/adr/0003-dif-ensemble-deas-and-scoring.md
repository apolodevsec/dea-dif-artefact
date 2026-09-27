# 3. Arquitetura do Componente Deep Isolation Forest (DIF) com DEAS e Representações Parametrizadas

Data: 2026-09-21

## Status

Aceito

## Contexto

O Módulo 4 opera sobre o espaço latente $z \in \mathbb{R}^9$ extraído pelo FC-DAE (Módulo 2 e 3) e deve gerar scores de anomalia robustos para tráfego de rede. Faz-se necessário definir a topologia das redes de projeção aleatórias $\Phi_i$, a estratégia de particionamento, a formalização do score DEAS em comparação ao Isolation Forest clássico e a metodologia de calibração de limiares.

## Decisão

1. **Topologia do Ensemble ($r$ vs. $t$):** Adotar parametrização via CLI (`--num_representations`) com valor padrão $r = 10$ redes de projeção aleatória, onde cada rede alimenta $t/r = 10$ árvores com subamostragens independentes ($\psi = 256$), totalizando $t = 100$ árvores. O modo estrito ($r = t = 100$) é mantido como opção de execução para análise comparativa de custo versus desempenho e alinhamento com a redação da Seção 2.4.1.
2. **Rede de Projeção ($\Phi_i$):** Implementar $\Phi_i$ como uma rede feedforward não-treinável em PyTorch com $L = 3$ camadas densas (parametrizável em $L \in [2, 4]$), ativação não-linear `LeakyReLU(negative_slope=0.2)` e dimensão de projeção $d = 9 = m$ (parametrizável para $d = 4 \approx m/2$). Os pesos são inicializados via Kaiming Normal e estritamente congelados (`requires_grad=False`).
3. **Implementação Nativa Modular:** Desenvolver o ensemble de isolamento e o cálculo DEAS de forma nativa em `src/models/dif.py` (PyTorch para projeção de representações em lotes e NumPy para construção e percurso da árvore), garantindo independência de dependências externas.
4. **Formulação do DEAS e Parâmetro $\lambda$:** O cálculo do comprimento de caminho ponderado por desvio adota:
   $$dev(x, T) = \frac{|z_j - v_j|}{\max(z_j) - \min(z_j) + \epsilon}$$
   $$h_{\text{DEAS}}(x) = \sum_{k=1}^{\text{prof}} \left(1 - \lambda \cdot dev_k(x)\right) + c(n_{\text{folha}})$$
   onde $\lambda \in [0, 1]$ é um hiperparâmetro explícito (padrão $\lambda = 0.5$). O documento técnico `dif.md` e o TCC registram essa equação como uma variante operacional inspirada em Xu et al. (2023). A propriedade matemática fundamental é mantida: quando $\lambda = 0$, o cálculo colapsa exatamente no Isolation Forest clássico ($h_{\text{DEAS}} = h$).
5. **Vetorização e Escalabilidade:** A inferência sobre 1,58 milhão de amostras combina projeção em lotes (`batch_size=32768`) via PyTorch, percurso de árvore vetorizado por máscaras booleanas por nível de profundidade e paralelismo de árvores via `joblib` (`n_jobs=-1`). Parâmetros de execução (hardware, batch size e n_jobs) serão registrados para garantir a reprodutibilidade da métrica de tempo de inferência.
6. **Regime de Treinamento e Calibração Dupla de Limiares:**
   - O ensemble é ajustado **exclusivamente sobre vetores latentes do tráfego legítimo normal**.
   - **Limiar não-supervisionado de referência ($\tau_{\text{DIF, ref}}$):** Calculado no Módulo 4 com base no percentil pré-fixado (ex.: $p=95$ ou $p=99$) sobre as amostras normais de validação.
   - **Ponto de operação final ($\tau_{\text{DIF, opt}}$):** Será determinado no Módulo 5 através da curva ROC e do Índice de Youden utilizando o conjunto de validação rotulado.
7. **Ablação Empírica do DEAS:** O script de inferência (`src/score_dif.py`) persistirá em `data/processed/dif_scores.parquet` tanto o score DEAS (`score_dif_deas`) quanto o score tradicional por comprimento de caminho (`score_dif_standard`), permitindo comprovar empiricamente a mitigação de *ghost regions* no Módulo 5.

## Consequências

### Positivas
- Redução substancial da sobrecarga de memória e tempo de inferência com $r=10$ sem perder a capacidade de rodar $r=100$.
- Controle matemático total e propriedade analítica limpa de ablação (quando $\lambda=0 \implies \text{iForest padrão}$).
- Distinção metodológica clara e rigorosa entre o limiar estatístico de referência não-supervisionado e a calibração com validação rotulada.

### Negativas
- Necessidade de atualizar e harmonizar a Tabela 3 e as Seções 2.4, 4.6 e 4.7 do texto do TCC para incluir os hiperparâmetros adicionais ($r$, $\lambda$, ativação, inicialização e percentil exato).

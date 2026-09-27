# Especificação Técnica: Módulo 4 — Componente Deep Isolation Forest (DIF)

## Problem Statement

Como pesquisador e desenvolvedor de sistemas de detecção de intrusões em redes (NIDS), preciso detectar ataques complexos e tráfego malicioso a partir de representações latentes comprimidas extraídas pelo Deep Autoencoder. O Isolation Forest clássico sofre com o problema das *ghost regions* (falsos positivos em regiões vazias com poucos dados) e limita-se a partições hiper-retangulares lineares paralelas aos eixos originais. Além disso, o cálculo tradicional baseado unicamente no comprimento do caminho discreto descarta a dispersão contínua e a densidade local dos nós. É necessário um mecanismo de isolamento não-linear, profundo e semanticamente enriquecido capaz de operar eficientemente sobre milhões de fluxos de rede sem assumir taxas artificiais de contaminação.

## Solution

Implementar o componente Deep Isolation Forest (DIF) como um ensemble de árvores de isolamento induzidas sobre subespaços gerados por redes neurais aleatórias não-lineares congeladas. O módulo realiza o particionamento sobre representações intermediárias e calcula pontuações de anomalia enriquecidas por desvio local (DEAS), atenuando *ghost regions* e capturando fronteiras não-lineares complexas. O componente opera sob regime estritamente não-supervisionado (ajustado unicamente sobre tráfego legítimo), fornece calibração de limiares estatísticos de referência e exporta em paralelo scores DEAS e scores tradicionais para viabilizar estudos de ablação empírica na avaliação experimental.

## User Stories

1. As a security researcher, I want the DIF model to be trained strictly on normal traffic latent representations, so that the anomaly detection model learns the legitimate manifold without contamination bias or label leakage.
2. As a security researcher, I want the ensemble to project latent vectors through non-linear random neural representations, so that the recursive partitioning can form arbitrary non-linear isolation boundaries.
3. As a developer, I want to configure the number of representation networks ($r$) independently from the total number of isolation trees ($t$), so that I can balance inferential throughput and representational diversity.
4. As a developer, I want to run the model in a strict mode where every tree has its own unique random neural network ($r = t$), so that I can validate the theoretical upper bound of projection diversity described in the literature.
5. As a researcher, I want the random neural networks to use frozen weights with fixed Kaiming initialization and LeakyReLU activation, so that the geometry is warped non-linearly without gradient saturation or parameter drift.
6. As a researcher, I want each isolation tree to be built on an independent subsample of 256 normal latent vectors, so that masking and swamping effects are mitigated while keeping training instantaneous.
7. As a security researcher, I want each internal node of an isolation tree to record the split attribute, cut value, and local data boundaries, so that continuous geometric deviations can be calculated during sample traversal.
8. As a security researcher, I want anomaly scoring to compute the DEAS metric using the relative distance to the cut value, so that points far from dense clusters receive higher anomaly weighting.
9. As a researcher, I want to configure the DEAS sensitivity factor ($\lambda$) via parameterization, so that I can tune the contribution of local deviation to the overall path length.
10. As an experimenter, I want the scoring function to collapse exactly to the standard Isolation Forest formula when $\lambda = 0$, so that I can execute clean mathematical and empirical ablation studies between standard iForest and DEAS.
11. As a data engineer, I want the scoring pipeline to evaluate latent vectors using batched neural forward passes and vectorized tree traversal, so that over 1.5 million network connections can be scored in minutes rather than hours.
12. As a systems engineer, I want multi-core CPU parallelism for tree evaluation, so that scoring scales linearly with available processor cores.
13. As an evaluator, I want the inference pipeline to output both the DEAS anomaly score and the standard path length score side by side in the output dataset, so that downstream evaluation can directly measure the impact of DEAS on false alarm reduction.
14. As an evaluator, I want the training process to calculate and persist an unsupervised reference threshold ($\tau_{\text{DIF, ref}}$) based on a specified validation percentile, so that baseline autonomous detection is possible without ground-truth attack labels.
15. As a machine learning practitioner, I want model artifacts, projection network weights, and tree structures to be cleanly serializable to disk, so that inference can be executed in separate stages or deployed independently.
16. As an experimenter, I want command-line interface arguments for tree count, subsample size, layer depth, projection dimension, representation count, and deviation factor, so that hyperparameter sweeps can be automated via scripts.
17. As an analyst, I want the inference output to retain original connection identifiers and ground-truth labels, so that downstream modules can generate ROC curves, PR curves, and confusion matrices without joining disparate tables.
18. As a developer, I want numerical stability protections (such as epsilon offsets) during division by node ranges, so that identical or near-constant features do not produce NaN scores or application crashes.
19. As a quality engineer, I want reproducible execution controlled by a global random seed, so that benchmarks, tree structures, and evaluation scores remain deterministic across runs.
20. As a research auditor, I want execution metadata (hardware specs, batch size, execution duration) recorded during scoring, so that reported inference latency metrics are fully reproducible.

## Implementation Decisions

- **Ensemble Topography and Representation Decoupling:** The model decouples the number of random neural representation networks from the total tree count. A pool of independent random projection networks is instantiated, and each network feeds a proportional subset of isolation trees. The strict one-network-per-tree configuration is supported through runtime configuration.
- **Random Neural Projector Design:** The projection networks are implemented as feedforward networks with frozen, non-trainable weights. The architecture supports configurable depth and projection dimension. Intermediate hidden layers use non-saturating non-linear activations (such as LeakyReLU with negative slope 0.2), while weights are initialized using orthogonal or variance-preserving random normal distributions.
- **Pure Native Python/NumPy Isolation Engine:** The isolation tree structure and traversal logic are built as a self-contained, native Python/NumPy module rather than wrapping black-box third-party isolation forest libraries. This guarantees direct access to node-level continuous statistics, bounding boxes, and split values required for DEAS scoring.
- **DEAS Mathematical Formulation:** Node traversal weights are modulated by continuous deviation ratios relative to local node bounds. The formulation guarantees strict mathematical backward-compatibility: setting the deviation scaling factor to zero reduces the path calculation strictly to the classical tree depth.
- **High-Throughput Vectorized Inference:** Inference splits operations into two stages: first, large-batch tensor matrix multiplications project the raw latent vectors into representation spaces; second, sample arrays traverse tree levels using boolean masking per depth layer. Tree scoring tasks are distributed across CPU workers using parallel process pools.
- **Dual Anomaly Metric Persistence:** Every inference pass computes and persists both the deviation-enhanced anomaly score and the standard path-length anomaly score. Both scores are bounded within the unit interval $[0, 1]$.
- **Separation of Unsupervised Reference and Supervised Operating Points:** The module strictly separates the unsupervised statistical reference threshold (computed from benign validation percentiles) from the supervised operating threshold (which will be optimized using labeled validation data and Youden's J statistic in the downstream evaluation phase).
- **Decoupled Workflow Scripts:** The workflow separates estimator definition, model training/persistence, and bulk inference scoring into distinct executable modules adhering to the project's single-responsibility pipeline structure.

## Testing Decisions

- **What Makes a Good Test:** Tests must evaluate external behavioral contracts and mathematical invariants through the public estimator interface, completely treating internal node recursion and weight matrices as private implementation details.
- **Modules Under Test:**
  - Estimator Core: Interface validation (`fit`, `score_samples`, serialization).
  - Mathematical Properties: Invariant verification that $\lambda = 0$ strictly equals standard Isolation Forest scoring.
  - Ghost Regions Mitigation: Verification that empty low-density regions between distant clusters are penalizados pelo DEAS com pontuações anômalas elevadas em comparação com o iForest tradicional (validação da hipótese central de atenuação de ghost regions).
  - Normal-Only Fit Isolation: Invariant verification that model fitting consumes strictly legitimate traffic samples without requiring or allowing contamination parameters or attack labels.
  - Lambda Monotonicity (Desirable): Verification that increasing $\lambda$ systematically scales the local deviation weight relative to tree depth.
  - Anomaly Sensitivity: Invariant verification that geometric outliers produce higher anomaly scores than centroid-adjacent points.
  - Numerical Stability: Zero-variance nodes, extreme values, and singleton splits must not yield NaNs or non-finite values.
  - Determinism: Identical random seeds must produce bit-exact scores.
  - End-to-End Pipeline: Small-scale synthetic latent parquet ingestion through training, model artifact generation, and inference scoring.

- **Prior Art in Codebase:** Follows the testing paradigms demonstrated in the autoencoder validation scripts (`validate_latent.py`) and dataset loader unit tests (`dataset.py`).

## Out of Scope

- Supervised hyperparameter tuning or operating threshold selection via ROC/Youden curves (reserved for Module 5).
- Multi-class intrusion classification or attack taxonomy categorization (the DIF component is strictly a binary anomaly scoring engine).
- Hybrid score combination with the Autoencoder reconstruction MSE (ensemble fusion is performed in Module 5).
- Real-time streaming packet capture or inline network socket sniffing.
- Modification of upstream FC-DAE architecture or regeneration of existing latent space files.

## Further Notes

- The latent dimensionality of the input data is $m = 9$, as established by the bottleneck of the trained FC-DAE.
- The default projection dimension $d$ matches the latent dimension ($d = m = 9$), with support for compression to $d = \lfloor m/2 \rfloor = 4$.
- The implementation directly respects ADR 0001, ADR 0002, and ADR 0003, and aligns with the domain terminology formalized in `CONTEXT.md`.

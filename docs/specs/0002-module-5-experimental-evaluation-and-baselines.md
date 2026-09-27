# Especificação Técnica: Módulo 5 — Protocolo de Validação Experimental e Avaliação de Baselines

## Problem Statement

Como pesquisador em cibersegurança e autor do TCC, preciso avaliar rigorosamente o framework híbrido proposto (FC-DAE + DIF com DEAS) em comparação com baselines representativos da literatura (Isolation Forest clássico, Autoencoder isolado, Random Forest supervisionado e LSTM Autoencoder recorrente). No entanto, o protocolo experimental anterior continha fragilidades metodológicas críticas: alocava 100% dos ataques no teste cego, inviabilizando a calibração legítima do limiar de operação por Youden na validação sem incorrer em vazamento de dados (*data leakage*); carecia de um baseline intermediário de ablação para comprovar o ganho isolado do DEAS em relação ao DIF clássico; não formalizava a fusão linear ponderada com reescalonamento robusto do erro MSE; e prometia métricas "multiclasse" completas para detectores que são, por definição teórica, detectores binários não-supervisionados de anomalia. É necessária uma infraestrutura de avaliação automatizada, modular e estatisticamente auditável que execute a cadeia de baselines, calcule o recall estratificado por tipo de ataque e conduza a análise de sensibilidade paramétrica.

## Solution

Implementar um pipeline unificado de avaliação experimental e calibração de baselines. O módulo estrutura o particionamento estratificado sem vazamento de dados (reservando 20% das anomalias na validação para calibração legítima de limiares e 80% no teste cego), implementa a cadeia comparativa de baselines em três degraus de ablação científica (incorporando o baseline B3b de DIF puro com $\lambda=0$ para isolar o impacto do DEAS), formaliza o detector híbrido por fusão linear ponderada com normalização robusta por percentis e regra disjuntiva (OR), gera métricas completas de detecção (Acurácia, Precisão, Recall, F1-Score, AUC-ROC, FPR, latência por fluxo e Recall estratificado por família de ataque) e automatiza a análise sistemática de sensibilidade via metodologia OFAT (*One-Factor-at-a-Time*).

## User Stories

1. As an experimental researcher, I want legitimate normal traffic partitioned into 70% train, 15% validation, and 15% blind test, so that training and validation remain strictly decoupled from final evaluation.
2. As an experimental researcher, I want anomalous attack traffic partitioned into 20% validation and 80% blind test with class stratification, so that operating thresholds can be calibrated via Youden's J statistic without test data leakage.
3. As an evaluator, I want rare attack classes (such as Heartbleed and Infiltration) to be flagged with statistical significance caveats in evaluation reports, so that low-sample variance is transparently communicated.
4. As an experimental auditor, I want a dedicated baseline evaluating classic Isolation Forest on raw features (B1), so that the baseline performance on unprocessed network traffic is established.
5. As an experimental auditor, I want a dedicated baseline evaluating the Deep Autoencoder reconstruction MSE alone (B2), so that the detection capability of reconstruction deviation is benchmarked in isolation.
6. As an experimental auditor, I want a dedicated baseline evaluating classic Isolation Forest on the compressed latent space (B3), so that the specific contribution of autoencoder feature extraction without deep isolation is quantified.
7. As a researcher, I want an explicit ablation baseline evaluating Deep Isolation Forest with $\lambda = 0$ on the latent space (B3b), so that the performance gain of non-linear random projections ($\Phi_i$) is isolated from the DEAS mechanism.
8. As a benchmark analyst, I want a supervised Random Forest baseline (B4), so that the theoretical performance ceiling of fully supervised tabular classification is documented.
9. As an evaluator, I want an LSTM Autoencoder recurrent baseline (B5) operating on sequential sliding windows, so that the comparison between temporal sequence modeling and tabular flow isolation is measured.
10. As an experimenter, I want the hybrid architecture to evaluate two distinct operational variants: weighted linear score fusion B6a ($\alpha \cdot s_{\text{DIF}} + (1 - \alpha) \cdot e_{\text{MSE, norm}}$) as the primary method, and the disjunctive OR rule B6b ($s_{\text{DIF}} > \tau_{\text{DIF}} \lor e_{\text{MSE}} > \tau_{\text{AE}}$) as the comparative veto variant.
11. As a developer, I want the reconstruction MSE normalized using validation-fitted robust percentiles ($p_1$ and $p_{99}$) with intentional saturation at 1.0, so that extreme reconstruction outliers cannot destabilize the linear combination.
12. As an evaluator, I want the alpha calibration optimizer to adaptively converge towards $\alpha \to 1.0$ when only DIF discriminates, and towards $\alpha \to 0.0$ when only MSE discriminates, proving true operational adaptivity.
13. As an experimental researcher, I want the optimal fusion weight $\alpha$ and decision threshold $\tau_{\text{opt}}$ calibrated exclusively on labeled validation data via ROC Youden optimization, so that the blind test set remains completely uncontaminated.
14. As an analyst, I want standard binary detection metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC, False Positive Rate) calculated over the blind test set, so that models can be compared against standard intrusion detection benchmarks.
15. As a network security practitioner, I want detection rate (Recall) stratified by specific attack category (such as DoS, PortScan, Brute Force, Web Attack, Botnet), so that coverage against distinct threat vectors is explicitly reported.
16. As an auditor, I want full multiclass precision, recall, and confusion matrices generated strictly for the supervised Random Forest, so that multinomial performance is reported only for models trained with multiclass capabilities.
17. As an efficiency benchmark analyst, I want average inference latency per sample measured and reported alongside detection metrics, so that real-time line-rate feasibility can be validated.
18. As an evaluator, I want the evaluator to quantitatively measure the ablation delta between B3b (DIF without DEAS) and B6a (DIF with DEAS), validating that the empirical gain in anomaly detection is attributable specifically to DEAS mitigation of ghost regions.
19. As an experimenter, I want systematic sensitivity analysis executed via One-Factor-at-a-Time (OFAT) across latent dimension $m$, network depth $L$, tree count $t$, and unsupervised percentile $p$, explicitly noting that $m = \lfloor n/8 \rfloor$ is the lower bound and that OFAT does not capture second-order parameter interactions.
20. As an operations engineer, I want evaluation results, calibration thresholds, and comparative tables exported to machine-readable JSON and formatted Markdown, with strictly deterministic stratified splitting governed by a random seed, so that experimental artifacts can be integrated directly into thesis tables and figures.

## Implementation Decisions

- **Stratified Leakage-Free Data Partitioner:** The data pipeline provides automated stratification that separates normal flows into training, validation, and testing sets, while allocating attack instances strictly across validation (for supervised calibration) and testing (for final evaluation). Stratification preserves minority class representations across partitions while raising informational flags for sample sizes below acceptable statistical thresholds.
- **Three-Tier Ablation Baseline Architecture:** The evaluation engine models comparisons across three sequential ablation tiers: first, comparing raw features versus compressed latent spaces; second, comparing classic axis-aligned trees versus random non-linear neural projections; and third, comparing standard path-length isolation versus deviation-enhanced scoring.
- **Robust Linear Score Fusion Engine:** The hybrid scoring module normalizes reconstruction errors by fitting lower and upper percentiles exclusively on validation data, applying strict unit-interval clipping to handle severe unseen test anomalies. The engine determines the optimal blending parameter $\alpha$ through bounded grid search maximizing the Youden index over validation ROC curves.
- **Parallel Disjunctive Decision Evaluator:** The system concurrently evaluates a dual-threshold disjunctive decision mechanism alongside linear score fusion. Each detector applies its independent validation-calibrated operating point, triggering an alert if either the deep isolation score or reconstruction error exceeds its corresponding boundary.
- **Stratified Attack-Class Sensitivity Calculator:** The evaluation framework separates binary anomaly scoring from attack-type attribution. For all unsupervised detectors, multiclass evaluation is formalized as stratified true positive rates computed across distinct ground-truth attack families, preventing conceptual conflation with multinomial classification.
- **Recurrent Baseline Sequential Windowing:** The sequential baseline constructs chronological sliding windows of consecutive flows, attributing the sequence reconstruction loss to the trailing connection. The implementation records that sequential modeling leverages lookback historical context absent from single-flow detectors.
- **OFAT Sensitivity Orchestrator:** The sensitivity evaluation module executes univariant sweeps starting from a fixed central baseline configuration. The orchestrator isolates parameter sweeps to avoid combinatorial grid explosion while documenting known trade-offs regarding parameter covariance.
- **Standardized Results Aggregation:** All evaluation metrics, confusion matrices, ROC coordinates, and calibration metadata are collected into standardized report structures supporting both structured JSON persistence and LaTeX/Markdown table generation.

## Testing Decisions

- **What Makes a Good Test:** Tests verify behavioral evaluation contracts and statistical invariants through the public evaluator interface, treating internal optimization loops and matrix multiplications as implementation details.
- **Modules Under Test:**
  - Partitioner Invariants: Verification that normal training sets contain zero attack samples and that validation and test partitions maintain mutually exclusive flow indices.
  - Stratified Split Determinism: Verification that identical random seeds produce bit-exact identical indices across train, validation, and test sets.
  - Leakage Guard: Verification that test samples are never observed during threshold calibration, alpha optimization, or Min-Max scaling parameter estimation.
  - Alpha Adaptivity Calibration Invariant: Verification that the alpha optimizer converges to $\alpha \to 1.0$ when only DIF has discriminative signal (MSE is random noise), and converges to $\alpha \to 0.0$ when only MSE has signal (DIF is random noise).
  - DEAS Ablation Delta Verification: Verification that B6a (with DEAS) achieves higher detection performance and ghost-region penalty than B3b ($\lambda = 0$), quantifying the empirical contribution of DEAS.
  - Mathematical Ablation Equivalence: Invariant verification that B3b ($\lambda = 0$) strictly matches unweighted tree traversal.
  - Schedular Boundedness: Verification that normalized reconstruction errors and fused hybrid scores remain strictly within the unit interval $[0, 1]$.
  - Stratified Recall Correctness: Verification that attack-family recall calculations match manual true positive tallies per class.
  - Latency Measurement Accuracy: Verification that telemetry accurately measures elapsed inference durations across sample batches.

- **Prior Art in Codebase:** Follows the testing architectures established in the latent validation suite (`tests/test_ghost_regions_and_scaling.py`) and estimator contract tests (`tests/test_dif.py`).

## Out of Scope

- Real-time packet parsing or kernel-level network interface card integration.
- Automated hyperparameter evolutionary algorithms (such as Bayesian optimization or genetic algorithms) beyond the planned OFAT sweep.
- Training new FC-DAE checkpoints for datasets other than CICIDS2017 within the current execution turn (architecture supports generic parquets, but multi-dataset execution is scheduled as future experiments).
- Interactive dashboard UI development (results are persisted as Markdown, JSON, and publication figures).

## Further Notes

- The primary evaluation runs over the 1,589,924 extracted connections of CICIDS2017 ($n=77, m=9$).
- The implementation directly honors ADR 0001, ADR 0002, ADR 0003, and ADR 0004, and aligns with the formal vocabulary defined in `CONTEXT.md`.

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Sequence
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.models.dif import DeepIsolationForest
from src.evaluation.evaluator import ExperimentalEvaluator
from src.evaluation.hybrid import HybridFusionDetector, _find_best_youden_threshold


class BaselineRunner:
    """Motor de orquestração e execução comparativa de baselines (B1 a B6b) e cálculo de ablação do DEAS."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        n_representations: int = 10,
        n_layers: int = 3,
        projection_dim: Optional[int] = None,
        lambda_deas: float = 0.5,
        p_low: float = 1.0,
        p_high: float = 99.0,
        n_grid: int = 101,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
        score_batch_size: int = 32768,
        reuse_forest_for_ablation: bool = True,
        max_supervised_rows: Optional[int] = None,
    ):
        self.score_batch_size = score_batch_size
        self.reuse_forest_for_ablation = reuse_forest_for_ablation
        self.max_supervised_rows = max_supervised_rows
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_representations = n_representations
        self.n_layers = n_layers
        self.projection_dim = projection_dim
        self.lambda_deas = lambda_deas
        self.p_low = p_low
        self.p_high = p_high
        self.n_grid = n_grid
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.evaluator = ExperimentalEvaluator()

    def run_all(
        self,
        X_train_raw: np.ndarray,
        Z_train: np.ndarray,
        mse_train: np.ndarray,
        X_val_raw: np.ndarray,
        Z_val: np.ndarray,
        mse_val: np.ndarray,
        y_val: np.ndarray,
        X_test_raw: np.ndarray,
        Z_test: np.ndarray,
        mse_test: np.ndarray,
        y_test: np.ndarray,
        test_classes: Optional[Sequence[Any]] = None,
        val_normal_mask: Optional[np.ndarray] = None,
        X_sup_train: Optional[np.ndarray] = None,
        y_sup_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Executa a cadeia completa de baselines tabulares e gera métricas comparativas padronizadas.

        :param X_sup_train: conjunto rotulado dedicado ao baseline supervisionado B4.
            Quando omitido, B4 recai sobre `X_train_raw` (assumido integralmente
            benigno) somado à validação rotulada.
        :param y_sup_train: rótulos binários correspondentes a `X_sup_train`.
        """
        n_test = len(y_test)
        summary_rows: Dict[str, Dict[str, float]] = {}
        stratified_recalls: Dict[str, Dict[str, Any]] = {}
        thresholds: Dict[str, float] = {}

        # -------------------------------------------------------------
        # B1: iForest isolado sobre features brutas
        # -------------------------------------------------------------
        b1_model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        b1_model.fit(X_train_raw)

        # Calibra o limiar na validação usando pontuação de anomalia invertida
        scores_b1_val = -b1_model.score_samples(X_val_raw)
        tau_b1, _ = _find_best_youden_threshold(y_val, scores_b1_val)
        thresholds["B1_iForest_Raw"] = tau_b1

        # Inferência no teste com telemetria de latência
        t0 = time.perf_counter()
        scores_b1_test = -b1_model.score_samples(X_test_raw)
        lat_b1 = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

        y_pred_b1 = (scores_b1_test > tau_b1).astype(int)
        metrics_b1 = self.evaluator.evaluate(
            y_test, y_pred_b1, scores=scores_b1_test, latency_ms_per_sample=lat_b1
        )
        summary_rows["B1_iForest_Raw"] = metrics_b1
        if test_classes is not None:
            stratified_recalls["B1_iForest_Raw"] = self.evaluator.stratified_recall(
                y_test, y_pred_b1, test_classes
            )

        # -------------------------------------------------------------
        # B2: Autoencoder isolado (Erro de Reconstrução MSE)
        # -------------------------------------------------------------
        tau_b2, _ = _find_best_youden_threshold(y_val, mse_val)
        thresholds["B2_Autoencoder_Alone"] = tau_b2

        y_pred_b2 = (mse_test > tau_b2).astype(int)
        # Latência considerada 0 ou computada no módulo 2
        metrics_b2 = self.evaluator.evaluate(
            y_test, y_pred_b2, scores=mse_test, latency_ms_per_sample=0.0
        )
        summary_rows["B2_Autoencoder_Alone"] = metrics_b2
        if test_classes is not None:
            stratified_recalls["B2_Autoencoder_Alone"] = self.evaluator.stratified_recall(
                y_test, y_pred_b2, test_classes
            )

        # -------------------------------------------------------------
        # B3: Pre-IF com iForest clássico sobre o espaço latente Z
        # -------------------------------------------------------------
        b3_model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        b3_model.fit(Z_train)

        scores_b3_val = -b3_model.score_samples(Z_val)
        tau_b3, _ = _find_best_youden_threshold(y_val, scores_b3_val)
        thresholds["B3_PreIF_iForest"] = tau_b3

        t0 = time.perf_counter()
        scores_b3_test = -b3_model.score_samples(Z_test)
        lat_b3 = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

        y_pred_b3 = (scores_b3_test > tau_b3).astype(int)
        metrics_b3 = self.evaluator.evaluate(
            y_test, y_pred_b3, scores=scores_b3_test, latency_ms_per_sample=lat_b3
        )
        summary_rows["B3_PreIF_iForest"] = metrics_b3
        if test_classes is not None:
            stratified_recalls["B3_PreIF_iForest"] = self.evaluator.stratified_recall(
                y_test, y_pred_b3, test_classes
            )

        # -------------------------------------------------------------
        # Ensemble DIF compartilhado entre B3b e B6a/B6b.
        #
        # A indução das árvores não consome lambda_deas: com a mesma semente, os
        # ensembles de lambda=0 e lambda=0.5 são estruturalmente idênticos. Reusar
        # uma única floresta torna a ablação do DEAS exata (a topologia é mantida
        # fixa e apenas o termo de desvio varia) e elimina um ajuste redundante.
        # `score_dif_standard` corresponde analiticamente a lambda = 0.
        # -------------------------------------------------------------
        b6_dif = DeepIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_representations=self.n_representations,
            n_layers=self.n_layers,
            projection_dim=self.projection_dim,
            lambda_deas=self.lambda_deas,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        t0 = time.perf_counter()
        b6_dif.fit(Z_train)
        dif_fit_seconds = time.perf_counter() - t0

        val_scores_dif = b6_dif.score_samples(Z_val, batch_size=self.score_batch_size)

        t0 = time.perf_counter()
        test_scores_dif = b6_dif.score_samples(Z_test, batch_size=self.score_batch_size)
        lat_dif_shared = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

        scores_dif_val = val_scores_dif["score_dif_deas"]
        scores_dif_test = test_scores_dif["score_dif_deas"]
        lat_b6_dif = lat_dif_shared

        # -------------------------------------------------------------
        # B3b: Pre-IF + DIF puro (lambda_deas = 0.0) sobre o espaço latente Z
        # Isola as projeções aleatórias não-lineares Phi_i do efeito do DEAS
        # -------------------------------------------------------------
        if self.reuse_forest_for_ablation:
            b3b_model = b6_dif
            scores_b3b_val = val_scores_dif["score_dif_standard"]
            scores_b3b_test = test_scores_dif["score_dif_standard"]
            # A travessia das árvores produz h_deas e h_standard no mesmo passo,
            # logo a latência medida é compartilhada entre B3b e B6.
            lat_b3b = lat_dif_shared
        else:
            b3b_model = DeepIsolationForest(
                n_estimators=self.n_estimators,
                max_samples=self.max_samples,
                n_representations=self.n_representations,
                n_layers=self.n_layers,
                projection_dim=self.projection_dim,
                lambda_deas=0.0,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
            b3b_model.fit(Z_train)

            scores_b3b_val = b3b_model.score_samples(Z_val, batch_size=self.score_batch_size)["score_dif_deas"]

            t0 = time.perf_counter()
            scores_b3b_test = b3b_model.score_samples(Z_test, batch_size=self.score_batch_size)["score_dif_deas"]
            lat_b3b = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

        tau_b3b, _ = _find_best_youden_threshold(y_val, scores_b3b_val)
        thresholds["B3b_PreIF_DIF_Pure"] = tau_b3b

        y_pred_b3b = (scores_b3b_test > tau_b3b).astype(int)
        metrics_b3b = self.evaluator.evaluate(
            y_test, y_pred_b3b, scores=scores_b3b_test, latency_ms_per_sample=lat_b3b
        )
        summary_rows["B3b_PreIF_DIF_Pure"] = metrics_b3b
        if test_classes is not None:
            stratified_recalls["B3b_PreIF_DIF_Pure"] = self.evaluator.stratified_recall(
                y_test, y_pred_b3b, test_classes
            )

        # -------------------------------------------------------------
        # B4: Random Forest supervisionado (teto teórico comparativo)
        # Treinado em `X_sup_train`/`y_sup_train` quando fornecidos (benigno de treino
        # + ataques de treino reservados); caso contrário, em treino normal + validação
        # rotulada. Em nenhuma hipótese há contato com o conjunto de teste cego.
        # -------------------------------------------------------------
        b4_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        if X_sup_train is not None and y_sup_train is not None:
            X_b4_train = np.asarray(X_sup_train)
            y_b4_train = np.asarray(y_sup_train, dtype=int)
        else:
            X_b4_train = np.vstack([X_train_raw, X_val_raw])
            y_b4_train = np.concatenate([np.zeros(len(X_train_raw), dtype=int), y_val])

        if self.max_supervised_rows is not None and len(X_b4_train) > self.max_supervised_rows:
            # Subamostragem estratificada por classe binária para manter o custo do
            # teto supervisionado tratável em datasets de milhões de fluxos.
            sub_rng = np.random.RandomState(self.random_state)
            fraction = self.max_supervised_rows / len(X_b4_train)
            keep_parts = []
            for cls in np.unique(y_b4_train):
                cls_idx = np.flatnonzero(y_b4_train == cls)
                quota = min(len(cls_idx), max(1, int(round(len(cls_idx) * fraction))))
                keep_parts.append(sub_rng.permutation(cls_idx)[:quota])
            keep = np.sort(np.concatenate(keep_parts))
            X_b4_train, y_b4_train = X_b4_train[keep], y_b4_train[keep]

        b4_rows_used = int(len(X_b4_train))
        b4_model.fit(X_b4_train, y_b4_train)

        t0 = time.perf_counter()
        probs_b4 = b4_model.predict_proba(X_test_raw)
        lat_b4 = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

        # Se o modelo viu apenas 1 classe (caso degenerado), probabilidade da anomalia é 0
        if probs_b4.shape[1] > 1:
            scores_b4_test = probs_b4[:, 1]
            y_pred_b4 = (scores_b4_test > 0.5).astype(int)
        else:
            scores_b4_test = np.zeros(n_test)
            y_pred_b4 = np.zeros(n_test, dtype=int)

        metrics_b4 = self.evaluator.evaluate(
            y_test, y_pred_b4, scores=scores_b4_test, latency_ms_per_sample=lat_b4
        )
        summary_rows["B4_RandomForest_Supervised"] = metrics_b4
        if test_classes is not None:
            stratified_recalls["B4_RandomForest_Supervised"] = self.evaluator.stratified_recall(
                y_test, y_pred_b4, test_classes
            )

        # -------------------------------------------------------------
        # B6a & B6b: Arquitetura Proposta Completa (FC-DAE + DIF com DEAS)
        # B6a: Fusão Linear Ponderada com calibração adaptativa de alfa
        # B6b: Regra Disjuntiva (OR)
        # -------------------------------------------------------------
        # O ensemble e os escores DEAS já foram calculados no bloco compartilhado.
        # Inicializa detector híbrido
        detector = HybridFusionDetector(
            p_low=self.p_low,
            p_high=self.p_high,
            n_grid=self.n_grid,
        )
        # Normalização com percentis baseados em tráfego normal legítimo
        if val_normal_mask is not None:
            detector.fit_scaling(mse_val[val_normal_mask])
        elif len(mse_train) > 0:
            detector.fit_scaling(mse_train)
        else:
            detector.fit_scaling(mse_val[y_val == 0])

        calib_res = detector.calibrate(scores_dif_val, mse_val, y_val)
        thresholds["B6a_Hybrid_Linear_tau"] = calib_res["tau_fused_opt"]
        thresholds["B6a_Hybrid_Linear_alpha"] = calib_res["alpha"]
        thresholds["B6b_Hybrid_OR_tau_dif"] = calib_res["tau_dif_opt"]
        thresholds["B6b_Hybrid_OR_tau_ae"] = calib_res["tau_ae_opt"]

        # Avaliação B6a (Linear)
        scores_b6a = detector.predict_proba(scores_dif_test, mse_test)
        y_pred_b6a = detector.predict(scores_dif_test, mse_test, mode="linear")
        metrics_b6a = self.evaluator.evaluate(
            y_test, y_pred_b6a, scores=scores_b6a, latency_ms_per_sample=lat_b6_dif
        )
        summary_rows["B6a_Hybrid_Linear"] = metrics_b6a
        if test_classes is not None:
            stratified_recalls["B6a_Hybrid_Linear"] = self.evaluator.stratified_recall(
                y_test, y_pred_b6a, test_classes
            )

        # Avaliação B6b (OR)
        scores_b6b = np.maximum(scores_dif_test, detector.normalize_mse(mse_test))
        y_pred_b6b = detector.predict(scores_dif_test, mse_test, mode="or")
        metrics_b6b = self.evaluator.evaluate(
            y_test, y_pred_b6b, scores=scores_b6b, latency_ms_per_sample=lat_b6_dif
        )
        summary_rows["B6b_Hybrid_OR"] = metrics_b6b
        if test_classes is not None:
            stratified_recalls["B6b_Hybrid_OR"] = self.evaluator.stratified_recall(
                y_test, y_pred_b6b, test_classes
            )

        # Montagem do DataFrame resumo
        summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")

        # Cálculo do ganho da ablação do DEAS: Delta_DEAS = F1(B6a) - F1(B3b)
        delta_deas_f1 = float(summary_df.loc["B6a_Hybrid_Linear", "f1_score"] - summary_df.loc["B3b_PreIF_DIF_Pure", "f1_score"])
        delta_deas_auc = float(summary_df.loc["B6a_Hybrid_Linear", "auc_roc"] - summary_df.loc["B3b_PreIF_DIF_Pure", "auc_roc"])

        return {
            "summary_df": summary_df,
            "thresholds": thresholds,
            "stratified_recall": stratified_recalls,
            "delta_deas_f1": delta_deas_f1,
            "delta_deas_auc": delta_deas_auc,
            "detector": detector,
            "b4_supervised_rows": b4_rows_used,
            "shared_dif_forest": bool(self.reuse_forest_for_ablation),
            "dif_fit_seconds": float(dif_fit_seconds),
            "scores": {
                "dif_val_deas": val_scores_dif["score_dif_deas"],
                "dif_val_standard": val_scores_dif["score_dif_standard"],
                "dif_test_deas": test_scores_dif["score_dif_deas"],
                "dif_test_standard": test_scores_dif["score_dif_standard"],
                "mse_test": np.asarray(mse_test, dtype=np.float64),
                "fused_test": scores_b6a,
            },
            # Escores contínuos de cada baseline no teste cego, na mesma ordem das
            # linhas de y_test. Viabilizam curvas ROC/PR sem reexecutar o treino.
            "test_scores_by_model": {
                "B1_iForest_Raw": np.asarray(scores_b1_test, dtype=np.float64),
                "B2_Autoencoder_Alone": np.asarray(mse_test, dtype=np.float64),
                "B3_PreIF_iForest": np.asarray(scores_b3_test, dtype=np.float64),
                "B3b_PreIF_DIF_Pure": np.asarray(scores_b3b_test, dtype=np.float64),
                "B4_RandomForest_Supervised": np.asarray(scores_b4_test, dtype=np.float64),
                "B6a_Hybrid_Linear": np.asarray(scores_b6a, dtype=np.float64),
                "B6b_Hybrid_OR": np.asarray(scores_b6b, dtype=np.float64),
            },
            "models": {
                "b1": b1_model,
                "b3": b3_model,
                "b3b": b3b_model,
                "b4": b4_model,
                "b6_dif": b6_dif,
            },
        }

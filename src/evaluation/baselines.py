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
        lambda_deas: float = 0.5,
        p_low: float = 1.0,
        p_high: float = 99.0,
        n_grid: int = 101,
        random_state: Optional[int] = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.n_representations = n_representations
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
    ) -> Dict[str, Any]:
        """Executa a cadeia completa de baselines tabulares e gera métricas comparativas padronizadas."""
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
        # B3b: Pre-IF + DIF puro (lambda_deas = 0.0) sobre o espaço latente Z
        # Isolando as projeções aleatórias não-lineares Phi_i do DEAS
        # -------------------------------------------------------------
        b3b_model = DeepIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_representations=self.n_representations,
            lambda_deas=0.0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        b3b_model.fit(Z_train)

        scores_b3b_val = b3b_model.score_samples(Z_val)["score_dif_deas"]
        tau_b3b, _ = _find_best_youden_threshold(y_val, scores_b3b_val)
        thresholds["B3b_PreIF_DIF_Pure"] = tau_b3b

        t0 = time.perf_counter()
        scores_b3b_test = b3b_model.score_samples(Z_test)["score_dif_deas"]
        lat_b3b = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

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
        # Treinado estritamente com dados de treino normal + validação rotulada
        # Zero contato com o conjunto de teste cego
        # -------------------------------------------------------------
        b4_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        X_b4_train = np.vstack([X_train_raw, X_val_raw])
        y_b4_train = np.concatenate([np.zeros(len(X_train_raw), dtype=int), y_val])
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
        b6_dif = DeepIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            n_representations=self.n_representations,
            lambda_deas=self.lambda_deas,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        b6_dif.fit(Z_train)

        scores_dif_val = b6_dif.score_samples(Z_val)["score_dif_deas"]

        # Inferência DIF no teste
        t0 = time.perf_counter()
        scores_dif_test = b6_dif.score_samples(Z_test)["score_dif_deas"]
        lat_b6_dif = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

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
            "models": {
                "b1": b1_model,
                "b3": b3_model,
                "b3b": b3b_model,
                "b4": b4_model,
                "b6_dif": b6_dif,
            },
        }

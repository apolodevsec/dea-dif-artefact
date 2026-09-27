import numpy as np
from typing import Optional, Sequence, Dict, Any
from sklearn.metrics import roc_curve


def _find_best_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Calcula o melhor limiar e o Índice de Youden correspondente (J = TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden_index = tpr - fpr
    best_idx = int(np.argmax(youden_index))
    best_tau = float(thresholds[best_idx])
    best_j = float(youden_index[best_idx])
    return best_tau, best_j


class HybridFusionDetector:
    """Detector de anomalias híbrido combinando escores do DIF e erro de reconstrução MSE."""

    def __init__(
        self,
        p_low: float = 1.0,
        p_high: float = 99.0,
        n_grid: int = 101,
    ):
        self.p_low = p_low
        self.p_high = p_high
        self.n_grid = n_grid

        self.p1_: float = 0.0
        self.p99_: float = 1.0
        self.alpha_: float = 0.5
        self.tau_fused_opt_: float = 0.5
        self.tau_dif_opt_: float = 0.5
        self.tau_ae_opt_: float = 0.001
        self.best_youden_: float = 0.0

    def fit_scaling(self, mse_normal: Sequence[float]) -> "HybridFusionDetector":
        """Calcula os percentis de corte robustos na validação normal."""
        arr = np.asarray(mse_normal, dtype=np.float64)
        self.p1_ = float(np.percentile(arr, self.p_low))
        self.p99_ = float(np.percentile(arr, self.p_high))
        if self.p99_ <= self.p1_:
            self.p99_ = self.p1_ + 1e-7
        return self

    def normalize_mse(self, mse: Sequence[float]) -> np.ndarray:
        """Normaliza o erro MSE para [0, 1] com saturação intencional (clipping)."""
        arr = np.asarray(mse, dtype=np.float64)
        scaled = (arr - self.p1_) / (self.p99_ - self.p1_)
        return np.clip(scaled, 0.0, 1.0)

    def calibrate(
        self,
        scores_dif_val: Sequence[float],
        mse_val: Sequence[float],
        y_val: Sequence[int],
    ) -> Dict[str, float]:
        """Otimiza alfa e limiares de decisão através da curva ROC na validação rotulada."""
        dif_arr = np.clip(np.asarray(scores_dif_val, dtype=np.float64), 0.0, 1.0)
        mse_norm_arr = self.normalize_mse(mse_val)
        y_arr = np.asarray(y_val, dtype=int)

        # 1. Calibrar os limiares individuais para a regra OR (B6b)
        self.tau_dif_opt_, _ = _find_best_youden_threshold(y_arr, dif_arr)
        # Para o AE, calibra sobre o MSE original
        raw_mse_arr = np.asarray(mse_val, dtype=np.float64)
        self.tau_ae_opt_, _ = _find_best_youden_threshold(y_arr, raw_mse_arr)

        # 2. Grid search sobre alpha in [0.0, 1.0] para maximizar Youden na Fusão Linear (B6a)
        alphas = np.linspace(0.0, 1.0, self.n_grid)
        best_alpha = 0.5
        best_j = -1.0
        best_auc = -1.0
        best_tau = 0.5

        from sklearn.metrics import roc_auc_score

        for a in alphas:
            fused = a * dif_arr + (1.0 - a) * mse_norm_arr
            tau, j = _find_best_youden_threshold(y_arr, fused)
            try:
                auc = float(roc_auc_score(y_arr, fused))
            except Exception:
                auc = 0.0

            # Maximiza primariamente o Youden J, depois AUC-ROC e por fim a margem de separação
            margin = float(np.mean(fused[y_arr == 1]) - np.mean(fused[y_arr == 0]))
            
            score_tuple = (round(j, 4), round(auc, 4), margin)
            best_tuple = (round(best_j, 4), round(best_auc, 4), getattr(self, "_best_margin", -999.0))

            if score_tuple > best_tuple:
                best_j = j
                best_auc = auc
                self._best_margin = margin
                best_alpha = float(a)
                best_tau = tau


        self.alpha_ = best_alpha
        self.tau_fused_opt_ = best_tau
        self.best_youden_ = best_j


        return {
            "alpha": self.alpha_,
            "tau_fused_opt": self.tau_fused_opt_,
            "tau_dif_opt": self.tau_dif_opt_,
            "tau_ae_opt": self.tau_ae_opt_,
            "best_youden": self.best_youden_,
        }

    def predict_proba(
        self,
        scores_dif: Sequence[float],
        mse: Sequence[float],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Gera a pontuação unificada contínua em [0, 1] via Fusão Linear Ponderada (B6a)."""
        dif_arr = np.clip(np.asarray(scores_dif, dtype=np.float64), 0.0, 1.0)
        mse_norm_arr = self.normalize_mse(mse)
        a = self.alpha_ if alpha is None else alpha
        fused = a * dif_arr + (1.0 - a) * mse_norm_arr
        return np.clip(fused, 0.0, 1.0)

    def predict(
        self,
        scores_dif: Sequence[float],
        mse: Sequence[float],
        mode: str = "linear",
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Classifica as amostras como anomalia (1) ou benigno (0)."""
        dif_arr = np.asarray(scores_dif, dtype=np.float64)
        mse_arr = np.asarray(mse, dtype=np.float64)

        if mode == "linear":
            probs = self.predict_proba(dif_arr, mse_arr)
            tau = self.tau_fused_opt_ if threshold is None else threshold
            return (probs > tau).astype(int)
        elif mode == "or":
            tau_dif = self.tau_dif_opt_
            tau_ae = self.tau_ae_opt_
            return ((dif_arr > tau_dif) | (mse_arr > tau_ae)).astype(int)
        else:
            raise ValueError(f"Modo desconhecido '{mode}'. Escolha 'linear' ou 'or'.")

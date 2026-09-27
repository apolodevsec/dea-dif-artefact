import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Sequence

from src.models.dif import DeepIsolationForest
from src.evaluation.evaluator import ExperimentalEvaluator
from src.evaluation.hybrid import _find_best_youden_threshold


class OFATSensitivityAnalyzer:
    """Analisador de Sensibilidade Univariada One-Factor-at-a-Time (OFAT) para hiperparâmetros do DIF."""

    DEFAULT_BASE_PARAMS = {
        "m": 9,
        "L": 3,
        "t": 100,
        "p": 95.0,
        "lambda_deas": 0.5,
        "n_representations": 10,
    }

    DEFAULT_SWEEPS = {
        "m": [9, 12, 19, 38],
        "L": [1, 2, 3, 4],
        "t": [50, 100, 200],
        "p": [90.0, 95.0, 97.0, 99.0],
    }

    def __init__(
        self,
        base_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.base_params = dict(self.DEFAULT_BASE_PARAMS)
        if base_params is not None:
            self.base_params.update(base_params)

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.evaluator = ExperimentalEvaluator()
        self.sweep_results_: Dict[str, pd.DataFrame] = {}

    def get_metadata(self) -> Dict[str, Any]:
        """Retorna metadados e ressalvas metodológicas do protocolo OFAT."""
        return {
            "base_params": self.base_params,
            "sweeps": self.DEFAULT_SWEEPS,
            "methodology_notes": (
                "A metodologia OFAT conduz varreduras univariadas a partir da configuração central de baseline. "
                "Vantagem: evita a explosão combinatória da grade O(K^4) mantendo custo O(4 * K). "
                "Ressalva assumida: a análise não captura second_order_interactions (covariâncias entre parâmetros, "
                "como a interação entre profundidade de representação L e dimensão latente m). "
                "Adicionalmente, m=9 constitui o limite inferior teórico floor(n/8) para n=77 do CICIDS2017."
            ),
        }

    def run_sweeps(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sweeps: Optional[Dict[str, List[Any]]] = None,
        output_figure_dir: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Executa as varreduras univariadas isoladas para m, L, t e p."""
        active_sweeps = sweeps if sweeps is not None else self.DEFAULT_SWEEPS
        results_by_param: Dict[str, pd.DataFrame] = {}

        n_test = len(y_test)

        for param_name, param_values in active_sweeps.items():
            records: List[Dict[str, Any]] = []

            for val in param_values:
                # Copia a configuração base e substitui o fator sob teste
                current_cfg = dict(self.base_params)
                current_cfg[param_name] = val

                # 1. Dimensão latente m
                m_dim = int(current_cfg["m"])
                # Garante que não exceda o número de colunas disponíveis
                eff_m = min(m_dim, X_train.shape[1])
                Z_train = X_train[:, :eff_m]
                Z_val = X_val[:, :eff_m]
                Z_test = X_test[:, :eff_m]

                # 2. Configura e ajusta o estimador DIF
                n_est = int(current_cfg["t"])
                n_lay = int(current_cfg["L"])
                n_rep = min(int(current_cfg.get("n_representations", 10)), n_est)
                lam = float(current_cfg.get("lambda_deas", 0.5))

                dif = DeepIsolationForest(
                    n_estimators=n_est,
                    max_samples=256,
                    n_representations=n_rep,
                    n_layers=n_lay,
                    lambda_deas=lam,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                dif.fit(Z_train)

                # 3. Inferência na validação para calibração
                scores_val = dif.score_samples(Z_val)["score_dif_deas"]

                # Limiar calibrado por Youden ou percentil p
                if param_name == "p":
                    # Varredura do percentil não-supervisionado sobre normais da validação
                    normal_scores = scores_val[y_val == 0]
                    if len(normal_scores) > 0:
                        tau = float(np.percentile(normal_scores, float(val)))
                    else:
                        tau, _ = _find_best_youden_threshold(y_val, scores_val)
                else:
                    tau, _ = _find_best_youden_threshold(y_val, scores_val)

                # 4. Inferência no teste com medição de latência
                t0 = time.perf_counter()
                scores_test = dif.score_samples(Z_test)["score_dif_deas"]
                lat_ms = ((time.perf_counter() - t0) * 1000.0) / max(1, n_test)

                y_pred = (scores_test > tau).astype(int)
                metrics = self.evaluator.evaluate(
                    y_test, y_pred, scores=scores_test, latency_ms_per_sample=lat_ms
                )

                row = {
                    "param_name": param_name,
                    "param_value": val,
                    "tau": tau,
                    **metrics,
                }
                records.append(row)

            df_param = pd.DataFrame(records)
            results_by_param[param_name] = df_param

        self.sweep_results_ = results_by_param

        # Exporta figuras se requisitado
        if output_figure_dir is not None:
            self._export_plots(output_figure_dir)

        return results_by_param

    def _export_plots(self, output_dir: str) -> None:
        """Exporta curvas de sensibilidade em formato gráfico."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            os.makedirs(output_dir, exist_ok=True)

            for param_name, df in self.sweep_results_.items():
                plt.figure(figsize=(7, 4.5))
                plt.plot(df["param_value"], df["f1_score"], marker="o", label="F1-Score", color="#1f77b4", lw=2)
                plt.plot(df["param_value"], df["auc_roc"], marker="s", label="AUC-ROC", color="#2ca02c", lw=2)
                plt.plot(df["param_value"], df["recall"], marker="^", label="Recall", color="#ff7f0e", lw=2)
                plt.plot(df["param_value"], df["fpr"], marker="x", label="FPR", color="#d62728", lw=1.5, ls="--")

                plt.title(f"OFAT Sensitivity Analysis: Parameter '{param_name}'", fontsize=12, fontweight="bold")
                plt.xlabel(f"Parameter Value ({param_name})", fontsize=10)
                plt.ylabel("Score [0, 1]", fontsize=10)
                plt.grid(True, linestyle=":", alpha=0.6)
                plt.legend(loc="best")
                plt.tight_layout()

                out_fig_path = os.path.join(output_dir, f"sensitivity_{param_name}.png")
                plt.savefig(out_fig_path, dpi=200)
                plt.close()
        except Exception:
            pass

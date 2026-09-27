import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ExperimentalEvaluator:
    """Avaliador unificado de métricas de detecção de intrusões (Tabela 4) e recall estratificado."""

    def evaluate(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        scores: Optional[Sequence[float]] = None,
        latency_ms_per_sample: Optional[float] = None,
    ) -> Dict[str, float]:
        """Calcula as métricas padronizadas binárias da Tabela 4 do TCC."""
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)

        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        auc = float("nan")
        if scores is not None and len(np.unique(y_true_arr)) > 1:
            try:
                auc = float(roc_auc_score(y_true_arr, scores))
            except Exception:
                pass

        results: Dict[str, float] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "fpr": fpr,
            "auc_roc": auc,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        }

        if latency_ms_per_sample is not None:
            results["latency_ms_per_sample"] = float(latency_ms_per_sample)

        return results

    def stratified_recall(
        self,
        y_true: Sequence[int],
        y_pred: Sequence[int],
        class_labels: Sequence[Any],
        normal_label: Any = "0",
    ) -> Dict[str, Dict[str, Any]]:
        """Calcula o Recall / Taxa de Detecção estratificada por família de ataque."""
        y_pred_arr = np.asarray(y_pred, dtype=int)
        class_labels_arr = np.asarray(class_labels, dtype=str)

        normal_str = str(normal_label).strip()
        classes = sorted(list(set(class_labels_arr)))

        stratified: Dict[str, Dict[str, Any]] = {}

        for cls in classes:
            cls_clean = cls.strip()
            # Pula a classe normal
            if (
                cls_clean == normal_str or
                cls_clean.upper() == "BENIGN" or
                cls_clean.upper() == "NORMAL"
            ):
                continue

            mask = class_labels_arr == cls
            total_instances = int(np.sum(mask))
            if total_instances == 0:
                continue

            detected = int(np.sum(y_pred_arr[mask] == 1))
            recall = float(detected / total_instances)

            stratified[cls] = {
                "count": total_instances,
                "detected": detected,
                "recall": recall,
            }

        return stratified

import numpy as np
import pandas as pd
import pytest
from src.evaluation.partitioner import DataPartitioner
from src.evaluation.evaluator import ExperimentalEvaluator


def test_data_partitioner_ratios_and_anti_leakage():
    """Valida o particionamento livre de vazamento (70/15/15 normais e 20/80 ataques)."""
    np.random.seed(42)
    n_normal = 1000
    n_dos = 200
    n_probe = 100

    df_normal = pd.DataFrame({"feat": np.random.randn(n_normal), "Label": ["BENIGN"] * n_normal})
    df_dos = pd.DataFrame({"feat": np.random.randn(n_dos), "Label": ["DoS"] * n_dos})
    df_probe = pd.DataFrame({"feat": np.random.randn(n_probe), "Label": ["Probe"] * n_probe})
    df_all = pd.concat([df_normal, df_dos, df_probe], ignore_index=True)

    partitioner = DataPartitioner(
        train_normal_ratio=0.70,
        val_normal_ratio=0.15,
        test_normal_ratio=0.15,
        val_attack_ratio=0.20,
        random_state=42,
    )
    partitions = partitioner.split(df_all, label_column="Label", normal_label="BENIGN")

    df_train = partitions["train"]
    df_val = partitions["val"]
    df_test = partitions["test"]

    # 1. Guarda Anti-Leakage: Treino tem estritamente 0% de ataques
    assert (df_train["Label"] == "BENIGN").all(), "Conjunto de treino não deve conter ataques!"
    assert len(df_train) == 700

    # 2. Partição das Normais: 700 treino, 150 val, 150 teste
    assert (df_val["Label"] == "BENIGN").sum() == 150
    assert (df_test["Label"] == "BENIGN").sum() == 150

    # 3. Partição dos Ataques (20% val / 80% teste com estratificação)
    assert (df_val["Label"] == "DoS").sum() == 40
    assert (df_test["Label"] == "DoS").sum() == 160
    assert (df_val["Label"] == "Probe").sum() == 20
    assert (df_test["Label"] == "Probe").sum() == 80

    # 4. Disjunção mútua de índices (sem interseção)
    idx_train = set(df_train.index)
    idx_val = set(df_val.index)
    idx_test = set(df_test.index)
    assert len(idx_train.intersection(idx_val)) == 0
    assert len(idx_train.intersection(idx_test)) == 0
    assert len(idx_val.intersection(idx_test)) == 0


def test_data_partitioner_determinism():
    """Garante reprodutibilidade determinística estrita para mesmas sementes."""
    df = pd.DataFrame({
        "feat": np.random.randn(500),
        "Label": ["BENIGN"] * 400 + ["Ataque"] * 100
    })

    part1 = DataPartitioner(random_state=999).split(df, label_column="Label")
    part2 = DataPartitioner(random_state=999).split(df, label_column="Label")

    pd.testing.assert_frame_equal(part1["train"], part2["train"])
    pd.testing.assert_frame_equal(part1["val"], part2["val"])
    pd.testing.assert_frame_equal(part1["test"], part2["test"])


def test_data_partitioner_rare_classes_flag():
    """Verifica que classes ultra-raras (< 15 amostras) são devidamente identificadas."""
    df = pd.DataFrame({
        "feat": np.random.randn(211),
        "Label": ["BENIGN"] * 200 + ["Heartbleed"] * 11
    })

    partitioner = DataPartitioner(random_state=42)
    partitions = partitioner.split(df, label_column="Label")
    rare_classes = partitions["rare_classes"]

    assert "Heartbleed" in rare_classes
    assert rare_classes["Heartbleed"] == 11


def test_evaluator_metrics_correctness():
    """Verifica a precisão matemática das métricas padronizadas da Tabela 4."""
    evaluator = ExperimentalEvaluator()
    # Cenário controlado:
    # 50 normais (0), 50 ataques (1)
    # y_pred: 40 TP, 10 FN, 45 TN, 5 FP
    y_true = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0] * 45 + [1] * 5 + [0] * 10 + [1] * 40)
    scores = np.where(y_pred == 1, 0.8, 0.2)

    res = evaluator.evaluate(y_true=y_true, y_pred=y_pred, scores=scores)

    # TP = 40, TN = 45, FP = 5, FN = 10
    # Acurácia = 85 / 100 = 0.85
    # Precisão = 40 / (40 + 5) = 40/45 = 0.8888...
    # Recall = 40 / (40 + 10) = 40/50 = 0.80
    # F1 = 2 * (40/45 * 40/50) / (40/45 + 40/50) = 80/95 = 0.8421...
    # FPR = 5 / (5 + 45) = 5/50 = 0.10
    assert pytest.approx(res["accuracy"], 0.001) == 0.85
    assert pytest.approx(res["precision"], 0.001) == 40 / 45
    assert pytest.approx(res["recall"], 0.001) == 0.80
    assert pytest.approx(res["f1_score"], 0.001) == 2 * (40 / 45 * 0.8) / (40 / 45 + 0.8)
    assert pytest.approx(res["fpr"], 0.001) == 0.10
    assert res["auc_roc"] > 0.80


def test_evaluator_stratified_recall():
    """Verifica a apuração de Recall individual por família de ataque."""
    evaluator = ExperimentalEvaluator()
    # 20 DoS (15 detectados -> Recall=0.75), 10 PortScan (10 detectados -> Recall=1.0)
    # 30 Benignos (28 TN, 2 FP)
    labels = ["BENIGN"] * 30 + ["DoS"] * 20 + ["PortScan"] * 10
    y_true = np.array([0] * 30 + [1] * 30)
    y_pred = np.array(
        [0] * 28 + [1] * 2 +          # Benignos: 28 TN, 2 FP
        [1] * 15 + [0] * 5 +          # DoS: 15 detectados, 5 não
        [1] * 10                      # PortScan: 10 detectados
    )

    stratified = evaluator.stratified_recall(y_true=y_true, y_pred=y_pred, class_labels=labels, normal_label="BENIGN")

    assert pytest.approx(stratified["DoS"]["recall"], 0.01) == 0.75
    assert stratified["DoS"]["count"] == 20
    assert stratified["DoS"]["detected"] == 15

    assert pytest.approx(stratified["PortScan"]["recall"], 0.01) == 1.00
    assert stratified["PortScan"]["count"] == 10
    assert stratified["PortScan"]["detected"] == 10

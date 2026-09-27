import numpy as np
import pandas as pd
import pytest
from src.evaluation.baselines import BaselineRunner


def test_baselines_execution_and_deas_ablation():
    """Valida a execução dos baselines tabulares (B1, B2, B3, B3b, B4, B6a, B6b) e cálculo da ablação do DEAS."""
    np.random.seed(42)
    n_train = 300
    n_val = 100
    n_test = 200
    n_features = 12
    m_latent = 4

    # Geração de dados sintéticos controlados
    # Treino normal compacto
    X_train_raw = np.random.normal(loc=0.0, scale=1.0, size=(n_train, n_features)).astype(np.float32)
    Z_train = X_train_raw[:, :m_latent]
    mse_train = np.random.uniform(0.0001, 0.002, size=n_train).astype(np.float32)

    # Validação (70 normais, 30 ataques)
    y_val = np.array([0] * 70 + [1] * 30)
    X_val_norm = np.random.normal(loc=0.0, scale=1.0, size=(70, n_features)).astype(np.float32)
    X_val_att = np.random.normal(loc=4.0, scale=1.5, size=(30, n_features)).astype(np.float32)
    X_val_raw = np.vstack([X_val_norm, X_val_att])
    Z_val = X_val_raw[:, :m_latent]
    mse_val = np.concatenate([
        np.random.uniform(0.0001, 0.002, size=70),
        np.random.uniform(0.005, 0.05, size=30)
    ]).astype(np.float32)

    # Teste (140 normais, 60 ataques: 40 DoS, 20 PortScan)
    y_test = np.array([0] * 140 + [1] * 60)
    classes_test = ["BENIGN"] * 140 + ["DoS"] * 40 + ["PortScan"] * 20
    X_test_norm = np.random.normal(loc=0.0, scale=1.0, size=(140, n_features)).astype(np.float32)
    X_test_dos = np.random.normal(loc=4.0, scale=1.5, size=(40, n_features)).astype(np.float32)
    X_test_scan = np.random.normal(loc=6.0, scale=1.0, size=(20, n_features)).astype(np.float32)
    X_test_raw = np.vstack([X_test_norm, X_test_dos, X_test_scan])
    Z_test = X_test_raw[:, :m_latent]
    mse_test = np.concatenate([
        np.random.uniform(0.0001, 0.002, size=140),
        np.random.uniform(0.005, 0.05, size=40),
        np.random.uniform(0.010, 0.08, size=20)
    ]).astype(np.float32)

    runner = BaselineRunner(random_state=42, n_jobs=1)

    results = runner.run_all(
        X_train_raw=X_train_raw,
        Z_train=Z_train,
        mse_train=mse_train,
        X_val_raw=X_val_raw,
        Z_val=Z_val,
        mse_val=mse_val,
        y_val=y_val,
        X_test_raw=X_test_raw,
        Z_test=Z_test,
        mse_test=mse_test,
        y_test=y_test,
        test_classes=classes_test,
    )

    summary_df = results["summary_df"]
    # Verifica que todos os baselines foram executados
    expected_baselines = ["B1_iForest_Raw", "B2_Autoencoder_Alone", "B3_PreIF_iForest", "B3b_PreIF_DIF_Pure", "B4_RandomForest_Supervised", "B6a_Hybrid_Linear", "B6b_Hybrid_OR"]
    for b in expected_baselines:
        assert b in summary_df.index, f"Baseline {b} não encontrado na tabela de resultados!"

    # Verifica o cálculo do delta de ablação do DEAS
    assert "delta_deas_f1" in results
    assert "delta_deas_auc" in results
    assert isinstance(results["delta_deas_f1"], float)

    # Verifica que métricas da Tabela 4 estão presentes
    for col in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "fpr"]:
        assert col in summary_df.columns

    # Verifica que stratified recall por ataque está computado
    assert "stratified_recall" in results
    assert "B6a_Hybrid_Linear" in results["stratified_recall"]
    assert "DoS" in results["stratified_recall"]["B6a_Hybrid_Linear"]
    assert "PortScan" in results["stratified_recall"]["B6a_Hybrid_Linear"]

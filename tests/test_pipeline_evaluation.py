import os
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.evaluation.sensitivity import OFATSensitivityAnalyzer
from src.baselines.lstm_ae import LSTMAutoencoderDetector


def test_ofat_sensitivity_analyzer():
    """Valida a execução sistemática da análise de sensibilidade univariada OFAT."""
    np.random.seed(42)
    n_train, n_val, n_test = 150, 60, 80
    n_features = 8

    X_train = np.random.normal(loc=0.0, scale=1.0, size=(n_train, n_features)).astype(np.float32)
    
    y_val = np.array([0] * 40 + [1] * 20)
    X_val = np.vstack([
        np.random.normal(loc=0.0, scale=1.0, size=(40, n_features)),
        np.random.normal(loc=3.0, scale=1.5, size=(20, n_features))
    ]).astype(np.float32)

    y_test = np.array([0] * 50 + [1] * 30)
    X_test = np.vstack([
        np.random.normal(loc=0.0, scale=1.0, size=(50, n_features)),
        np.random.normal(loc=3.5, scale=1.5, size=(30, n_features))
    ]).astype(np.float32)

    analyzer = OFATSensitivityAnalyzer(
        base_params={"m": 4, "L": 2, "t": 20, "p": 95.0, "lambda_deas": 0.5},
        random_state=42,
        n_jobs=1,
    )

    # Mini sweep rápido para validação unitária
    custom_sweeps = {
        "m": [4, 6],
        "L": [2, 3],
        "t": [10, 20],
        "p": [90.0, 95.0],
    }

    sweep_results = analyzer.run_sweeps(
        X_train=X_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        sweeps=custom_sweeps,
    )

    assert "m" in sweep_results
    assert "L" in sweep_results
    assert "t" in sweep_results
    assert "p" in sweep_results

    # Verifica colunas e tipos
    for param_name in ["m", "L", "t", "p"]:
        df = sweep_results[param_name]
        assert isinstance(df, pd.DataFrame)
        assert "param_value" in df.columns
        assert "f1_score" in df.columns
        assert "auc_roc" in df.columns
        assert len(df) == 2

    # Verifica documentação de ressalva teórica do OFAT
    assert "methodology_notes" in analyzer.get_metadata()
    assert "second_order_interactions" in analyzer.get_metadata()["methodology_notes"]


def test_lstm_ae_recurrent_baseline():
    """Valida o baseline sequencial recorrente B5 (LSTM-AE) com janelas deslizantes."""
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    seq_len = 5
    n_features = 4
    n_train_seq = 60
    n_val_seq = 30
    n_test_seq = 40

    # Dados temporais normais
    X_train_seq = np.random.normal(0.0, 0.5, size=(n_train_seq, seq_len, n_features)).astype(np.float32)

    # Validação (20 normais, 10 ataques com pico)
    y_val = np.array([0] * 20 + [1] * 10)
    X_val_norm = np.random.normal(0.0, 0.5, size=(20, seq_len, n_features)).astype(np.float32)
    X_val_att = np.random.normal(3.0, 1.0, size=(10, seq_len, n_features)).astype(np.float32)
    X_val_seq = np.vstack([X_val_norm, X_val_att])

    # Teste (25 normais, 15 ataques)
    y_test = np.array([0] * 25 + [1] * 15)
    X_test_norm = np.random.normal(0.0, 0.5, size=(25, seq_len, n_features)).astype(np.float32)
    X_test_att = np.random.normal(3.5, 1.0, size=(15, seq_len, n_features)).astype(np.float32)
    X_test_seq = np.vstack([X_test_norm, X_test_att])

    model = LSTMAutoencoderDetector(
        input_dim=n_features,
        hidden_dim=8,
        latent_dim=4,
        epochs=3,
        batch_size=16,
        lr=0.01,
        random_state=42,
    )

    model.fit(X_train_seq)

    # Calibração Youden na validação
    calib = model.calibrate(X_val_seq, y_val)
    assert "tau_opt" in calib
    assert "best_youden" in calib

    # Predição e avaliação no teste
    scores_test = model.score_samples(X_test_seq)
    y_pred_test = model.predict(X_test_seq)

    assert len(scores_test) == n_test_seq
    assert len(y_pred_test) == n_test_seq
    assert np.all((y_pred_test == 0) | (y_pred_test == 1))
    # Discriminação: média dos escores dos ataques deve ser superior aos normais
    assert np.mean(scores_test[y_test == 1]) > np.mean(scores_test[y_test == 0])


def test_evaluate_protocol_pipeline(tmp_path: Path):
    """Valida a execução ponta a ponta do pipeline com exportação de relatórios Markdown e JSON."""
    from src.evaluate_protocol import run_evaluation_pipeline

    # Cria diretório temporário para saídas
    output_dir = tmp_path / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "experimental_results.json"
    md_path = output_dir / "experimental_results.md"

    # Executa em modo sintético reduzido
    summary = run_evaluation_pipeline(
        synthetic=True,
        n_samples=500,
        n_features=12,
        output_dir=str(output_dir),
        random_state=42,
        run_ofat=False,
    )

    assert json_path.exists(), "experimental_results.json não foi gerado!"
    assert md_path.exists(), "experimental_results.md não foi gerado!"

    # Inspeciona o JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "baselines" in data
    assert "delta_deas_f1" in data
    assert "delta_deas_auc" in data
    assert "stratified_recall" in data
    assert "calibration" in data

    # Inspeciona o Markdown
    content = md_path.read_text(encoding="utf-8")
    assert "## Tabela 4: Métricas Comparativas de Baselines" in content
    assert "B6a_Hybrid_Linear" in content
    assert "B3b_PreIF_DIF_Pure" in content
    assert "Ablação do DEAS" in content

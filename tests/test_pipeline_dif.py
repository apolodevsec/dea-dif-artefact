import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest


def test_cli_pipeline_end_to_end(tmp_path):
    """Teste de integração ponta a ponta dos scripts src/train_dif.py e src/score_dif.py."""
    np.random.seed(42)

    # 1. Gerar dataset sintético simulando data/processed/latent_space.parquet
    n_normal = 300
    n_attack = 50
    m = 9

    # Tráfego normal compacto em torno de 0.0
    Z_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, m)).astype(np.float32)
    # Tráfego de ataque disperso em torno de 6.0
    Z_attack = np.random.normal(loc=6.0, scale=1.5, size=(n_attack, m)).astype(np.float32)

    Z_all = np.vstack([Z_normal, Z_attack])
    labels = ["0"] * n_normal + ["DDoS-Attack"] * n_attack
    mse_vals = np.random.uniform(0.0001, 0.005, size=len(Z_all)).astype(np.float32)

    z_cols = [f"z_{i}" for i in range(m)]
    df_synthetic = pd.DataFrame(Z_all, columns=z_cols)
    df_synthetic["mse"] = mse_vals
    df_synthetic["Label"] = labels

    parquet_in = tmp_path / "latent_space.parquet"
    df_synthetic.to_parquet(parquet_in, index=True)

    model_out = tmp_path / "dif_ensemble.joblib"
    metadata_out = tmp_path / "dif_metadata.json"
    scores_out = tmp_path / "dif_scores.parquet"

    python_exe = sys.executable

    # 2. Executar src/train_dif.py
    cmd_train = [
        python_exe,
        "src/train_dif.py",
        "--latent_parquet", str(parquet_in),
        "--model_out", str(model_out),
        "--metadata_out", str(metadata_out),
        "--trees", "15",
        "--subsample", "64",
        "--representations", "3",
        "--val_percentile", "95.0",
        "--n_jobs", "1",
    ]
    res_train = subprocess.run(cmd_train, capture_output=True, text=True)
    assert res_train.returncode == 0, f"train_dif falhou:\n{res_train.stderr}\n{res_train.stdout}"
    assert os.path.exists(model_out)
    assert os.path.exists(metadata_out)

    with open(metadata_out, "r") as f:
        meta = json.load(f)
    assert "tau_dif_ref_deas" in meta
    assert "tau_dif_ref_standard" in meta
    assert meta["n_samples_train_normal"] > 0

    # 3. Executar src/score_dif.py
    cmd_score = [
        python_exe,
        "src/score_dif.py",
        "--latent_parquet", str(parquet_in),
        "--model_path", str(model_out),
        "--metadata_path", str(metadata_out),
        "--output_parquet", str(scores_out),
        "--batch_size", "128",
        "--n_jobs", "1",
    ]
    res_score = subprocess.run(cmd_score, capture_output=True, text=True)
    assert res_score.returncode == 0, f"score_dif falhou:\n{res_score.stderr}\n{res_score.stdout}"
    assert os.path.exists(scores_out)

    # 4. Validar o parquet de scores gerado
    df_scores = pd.read_parquet(scores_out)
    assert len(df_scores) == n_normal + n_attack
    assert "score_dif_deas" in df_scores.columns
    assert "score_dif_standard" in df_scores.columns
    assert "label_binary" in df_scores.columns
    assert "label_multiclass" in df_scores.columns
    assert "mse" in df_scores.columns

    # Rótulo binário: normal = 0, ataque = 1
    assert (df_scores["label_binary"].iloc[:n_normal] == 0).all()
    assert (df_scores["label_binary"].iloc[n_normal:] == 1).all()

    # Média de scores dos ataques deve ser substancialmente superior à dos normais
    mean_attack_score = df_scores.iloc[n_normal:]["score_dif_deas"].mean()
    mean_normal_score = df_scores.iloc[:n_normal]["score_dif_deas"].mean()
    assert mean_attack_score > mean_normal_score + 0.10

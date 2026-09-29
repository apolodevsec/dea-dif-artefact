import json

import numpy as np
import pytest

from src.train_pipeline import main
from tests.test_nids_datasets import write_dataset


@pytest.fixture()
def three_datasets(tmp_path):
    """Reproduz os três artefatos processados, cada um com seu esquema de rótulo."""
    write_dataset(
        tmp_path / "NSL-KDD-processados",
        label_column="xAttack",
        normal_label="Normal",
        attack_counts={"DoS": 120, "Probe": 50, "U2R": 4},
        n_normal_train=320,
        n_normal_test=160,
        seed=11,
    )
    write_dataset(
        tmp_path / "UNSW-NB15-processados",
        label_column="attack_cat",
        normal_label="Normal",
        attack_counts={"Exploits": 100, "Fuzzers": 60, "Worms": 5},
        n_normal_train=300,
        n_normal_test=150,
        seed=22,
    )
    write_dataset(
        tmp_path / "CICIDS2017-processados",
        label_column="Label",
        normal_label="BENIGN",
        attack_counts={"DoS Hulk": 110, "Web Attack � XSS": 40, "Heartbleed": 3},
        n_normal_train=300,
        n_normal_test=150,
        seed=33,
    )
    return tmp_path


def _fast_args(data_root, output_dir, models_dir, extra=None):
    args = [
        "--data_root", str(data_root),
        "--output_dir", str(output_dir),
        "--models_dir", str(models_dir),
        "--epochs", "4",
        "--batch_size", "64",
        "--lr", "5e-3",
        "--trees", "12",
        "--subsample", "64",
        "--representations", "3",
        "--alpha_grid", "21",
        "--n_jobs", "1",
        "--max_supervised_rows", "0",
    ]
    return args + (extra or [])


def test_pipeline_runs_all_three_datasets(three_datasets, tmp_path):
    output_dir = tmp_path / "reports"
    payload = main(_fast_args(three_datasets, output_dir, tmp_path / "models"))

    assert payload["failures"] == []
    assert len(payload["datasets"]) == 3
    assert {d["dataset"] for d in payload["datasets"]} == {"NSL-KDD", "UNSW-NB15", "CICIDS2017"}

    # Relatórios consolidados
    assert (output_dir / "consolidated_results.md").exists()
    assert (output_dir / "consolidated_results.json").exists()
    consolidated = (output_dir / "consolidated_results.md").read_text(encoding="utf-8")
    for dataset in ("NSL-KDD", "UNSW-NB15", "CICIDS2017"):
        assert dataset in consolidated
    assert "Ablação do DEAS por Dataset" in consolidated

    expected_baselines = {
        "B1_iForest_Raw",
        "B2_Autoencoder_Alone",
        "B3_PreIF_iForest",
        "B3b_PreIF_DIF_Pure",
        "B4_RandomForest_Supervised",
        "B6a_Hybrid_Linear",
        "B6b_Hybrid_OR",
    }

    for result in payload["datasets"]:
        slug = result["dataset"]
        dataset_dir = output_dir / slug
        assert (dataset_dir / "experimental_results.md").exists()
        assert (dataset_dir / "experimental_results.json").exists()
        assert (dataset_dir / "fc_dae_training_history.json").exists()
        assert (tmp_path / "models" / slug / "fc_dae_best.pt").exists()

        assert expected_baselines <= set(result["baselines"])

        # Módulo 2: bottleneck e histórico coerentes
        autoencoder = result["autoencoder"]
        assert autoencoder["bottleneck_dim"] >= 1
        assert autoencoder["input_dim"] == result["partitions"]["n_features"]
        assert len(autoencoder["history"]) == autoencoder["epochs_run"]

        # Módulo 4: limiar não-supervisionado dentro do domínio do escore
        assert 0.0 <= result["dif"]["tau_dif_ref_deas"] <= 1.0
        assert result["dif"]["projection_dim"] == autoencoder["bottleneck_dim"]

        # Calibração supervisionada existe e alfa é válido
        supervised = result["calibration"]["supervised"]
        assert 0.0 <= supervised["B6a_Hybrid_Linear_alpha"] <= 1.0

        # Métricas no domínio esperado
        for metrics in result["baselines"].values():
            for key in ("accuracy", "precision", "recall", "f1_score", "fpr"):
                assert 0.0 <= metrics[key] <= 1.0

        # Recall estratificado cobre as famílias de ataque, nunca a classe benigna
        strat = result["stratified_recall"]["B6a_Hybrid_Linear"]
        assert strat
        assert "BENIGN" not in strat and "Normal" not in strat

        markdown = (dataset_dir / "experimental_results.md").read_text(encoding="utf-8")
        assert "## 4. Tabela 4 — Métricas Comparativas no Teste Cego" in markdown
        assert "## 5. Ablação do DEAS" in markdown
        assert "tau_AE" in markdown


def test_pipeline_flags_rare_classes_and_normalizes_cicids_labels(three_datasets, tmp_path):
    output_dir = tmp_path / "reports"
    payload = main(
        _fast_args(three_datasets, output_dir, tmp_path / "models", ["--datasets", "CICIDS2017"])
    )

    assert len(payload["datasets"]) == 1
    result = payload["datasets"][0]
    assert result["dataset"] == "CICIDS2017"

    families = set(result["stratified_recall"]["B6a_Hybrid_Linear"])
    assert "Web Attack - XSS" in families
    assert not any("�" in f for f in families)

    # Heartbleed tem 3 instâncias no treino e 2 no teste -> marcada como rara
    assert "Heartbleed" in result["rare_classes"]
    markdown = (output_dir / "CICIDS2017" / "experimental_results.md").read_text(encoding="utf-8")
    assert "Ressalva estatística de classes raras" in markdown


def test_pipeline_selects_single_dataset_dir_and_saves_latent(three_datasets, tmp_path):
    output_dir = tmp_path / "reports"
    payload = main(
        [
            "--dataset_dir", str(three_datasets / "NSL-KDD-processados"),
            "--output_dir", str(output_dir),
            "--models_dir", str(tmp_path / "models"),
            "--epochs", "3",
            "--batch_size", "64",
            "--trees", "8",
            "--subsample", "64",
            "--representations", "2",
            "--alpha_grid", "11",
            "--n_jobs", "1",
            "--save_latent",
        ]
    )

    assert len(payload["datasets"]) == 1
    latent_test = output_dir / "NSL-KDD" / "latent_space_test.parquet"
    assert latent_test.exists()

    import pandas as pd

    frame = pd.read_parquet(latent_test)
    m = payload["datasets"][0]["autoencoder"]["bottleneck_dim"]
    assert list(frame.columns) == [f"z_{i}" for i in range(m)] + ["mse", "xAttack"]
    assert len(frame) == payload["datasets"][0]["partitions"]["sizes"]["test_total"]


def test_pipeline_reports_failure_without_aborting_other_datasets(three_datasets, tmp_path):
    """Um dataset inválido é registrado em `failures` sem interromper os demais."""
    broken = three_datasets / "BROKEN-processados"
    broken.mkdir()
    for name in ("X_train_baseline", "y_train_baseline", "X_test", "y_test"):
        (broken / f"{name}.parquet").write_bytes(b"nao-e-um-parquet")

    payload = main(
        _fast_args(three_datasets, tmp_path / "reports", tmp_path / "models", ["--datasets", "BROKEN,NSL-KDD"])
    )

    assert len(payload["failures"]) == 1
    assert "BROKEN" in payload["failures"][0]["dataset_dir"]
    assert len(payload["datasets"]) == 1
    assert payload["datasets"][0]["dataset"] == "NSL-KDD"


def test_ofat_sweep_is_optional_and_records_all_factors(three_datasets, tmp_path):
    output_dir = tmp_path / "reports"
    payload = main(
        [
            "--dataset_dir", str(three_datasets / "UNSW-NB15-processados"),
            "--output_dir", str(output_dir),
            "--models_dir", str(tmp_path / "models"),
            "--epochs", "2",
            "--batch_size", "64",
            "--trees", "6",
            "--subsample", "32",
            "--representations", "2",
            "--alpha_grid", "11",
            "--n_jobs", "1",
            "--run_ofat",
        ]
    )

    ofat = payload["datasets"][0]["ofat"]
    assert set(ofat["sweeps"]) == {"L", "t", "p", "lambda_deas"}
    for records in ofat["sweeps"].values():
        assert records
        for record in records:
            assert "f1_score" in record and "auc_roc" in record
    assert "second_order_interactions" in ofat["metadata"]["methodology_notes"]

    markdown = (output_dir / "UNSW-NB15" / "experimental_results.md").read_text(encoding="utf-8")
    assert "## 8. Análise de Sensibilidade OFAT" in markdown


def test_latent_dim_sweep_retrains_autoencoder(three_datasets, tmp_path):
    output_dir = tmp_path / "reports"
    payload = main(
        [
            "--dataset_dir", str(three_datasets / "NSL-KDD-processados"),
            "--output_dir", str(output_dir),
            "--models_dir", str(tmp_path / "models"),
            "--epochs", "2",
            "--batch_size", "64",
            "--trees", "6",
            "--subsample", "32",
            "--representations", "2",
            "--alpha_grid", "11",
            "--n_jobs", "1",
            "--ofat_m_values", "2", "4",
            "--ofat_m_epochs", "2",
        ]
    )

    records = payload["datasets"][0]["ofat"]["m_retrained"]
    assert [r["param_value"] for r in records] == [2, 4]
    for record in records:
        assert record["param_name"] == "m"
        assert np.isfinite(record["ae_best_val_loss"])
    assert (output_dir / "NSL-KDD" / "ofat_latent_dim_sweep.json").exists()


def test_missing_data_source_is_rejected():
    with pytest.raises(SystemExit):
        main(["--epochs", "1"])

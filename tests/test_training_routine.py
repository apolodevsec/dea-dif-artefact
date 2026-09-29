import json
import math

import numpy as np
import pytest
import torch

from src.models.autoencoder import FCDAE
from src.training.ae_trainer import (
    AETrainConfig,
    default_bottleneck_dim,
    save_training_history,
    train_fc_dae,
)
from src.training.calibration import (
    calibrate_ae_thresholds,
    mu_sigma_threshold,
    percentile_threshold,
)
from src.training.latent import extract_latent_and_mse, latent_frame


def make_benign(n, n_features=12, seed=0):
    """Tráfego benigno sintético em [0, 1] com estrutura de baixa dimensão aprendível."""
    rng = np.random.RandomState(seed)
    base = rng.uniform(0.2, 0.6, size=(n, 3))
    mix = rng.uniform(0.0, 1.0, size=(3, n_features))
    return np.clip(base @ mix / 3.0 + rng.normal(0, 0.01, size=(n, n_features)), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# Treino do FC-DAE
# --------------------------------------------------------------------------
def test_default_bottleneck_follows_framework_rule():
    assert default_bottleneck_dim(77) == 9  # CICIDS2017: floor(n/8)
    assert default_bottleneck_dim(115) == 14  # NSL-KDD
    assert default_bottleneck_dim(194) == 24  # UNSW-NB15
    assert default_bottleneck_dim(20) == 5  # n < 50 -> floor(n/4)
    assert default_bottleneck_dim(1) == 1


def test_train_fc_dae_reduces_validation_loss(tmp_path):
    X_train, X_val = make_benign(600, seed=1), make_benign(200, seed=2)
    checkpoint = tmp_path / "fc_dae_best.pt"

    result = train_fc_dae(
        X_train,
        X_val,
        config=AETrainConfig(epochs=12, batch_size=128, lr=5e-3, seed=42, verbose=False),
        checkpoint_path=str(checkpoint),
        feature_columns=[f"f{i}" for i in range(X_train.shape[1])],
    )

    assert result.epochs_run == 12
    assert result.bottleneck_dim == default_bottleneck_dim(X_train.shape[1])
    assert math.isfinite(result.best_val_loss)
    # A última época deve ser claramente melhor que a primeira.
    assert result.history[-1]["val_loss"] < result.history[0]["val_loss"]
    # O melhor valor registrado coincide com o mínimo do histórico.
    assert result.best_val_loss == pytest.approx(min(h["val_loss"] for h in result.history))
    assert checkpoint.exists()


def test_checkpoint_round_trip_restores_best_epoch(tmp_path):
    X_train, X_val = make_benign(400, seed=3), make_benign(150, seed=4)
    checkpoint = tmp_path / "ckpt.pt"
    columns = [f"f{i}" for i in range(X_train.shape[1])]

    result = train_fc_dae(
        X_train,
        X_val,
        config=AETrainConfig(epochs=6, batch_size=128, seed=42, verbose=False),
        checkpoint_path=str(checkpoint),
        feature_columns=columns,
        extra_checkpoint_fields={"dataset": "SYNTH"},
    )

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    assert payload["input_dim"] == X_train.shape[1]
    assert payload["bottleneck_dim"] == result.bottleneck_dim
    assert payload["feature_columns"] == columns
    assert payload["dataset"] == "SYNTH"
    assert payload["best_epoch"] == result.best_epoch

    restored = FCDAE(input_dim=payload["input_dim"], bottleneck_dim=payload["bottleneck_dim"])
    restored.load_state_dict(payload["model_state_dict"])
    restored.eval()

    # O modelo em memória já traz os pesos da melhor época: ambos devem coincidir.
    Z_a, mse_a = extract_latent_and_mse(result.model, X_val, batch_size=64)
    Z_b, mse_b = extract_latent_and_mse(restored, X_val, batch_size=64)
    np.testing.assert_allclose(Z_a, Z_b, atol=1e-6)
    np.testing.assert_allclose(mse_a, mse_b, atol=1e-9)


def test_early_stopping_triggers_on_stagnation():
    X_train, X_val = make_benign(200, seed=5), make_benign(80, seed=6)
    result = train_fc_dae(
        X_train,
        X_val,
        # lr nula impede qualquer melhora: a paciência deve encerrar o treino.
        config=AETrainConfig(epochs=30, batch_size=64, lr=0.0, patience=2, seed=42, verbose=False),
    )
    assert result.early_stopped is True
    assert result.epochs_run < 30


def test_training_is_deterministic_for_a_fixed_seed():
    X_train, X_val = make_benign(300, seed=7), make_benign(100, seed=8)
    config = AETrainConfig(epochs=4, batch_size=64, seed=1234, verbose=False)

    a = train_fc_dae(X_train, X_val, config=config)
    b = train_fc_dae(X_train, X_val, config=config)
    assert a.best_val_loss == pytest.approx(b.best_val_loss, rel=1e-9)

    c = train_fc_dae(X_train, X_val, config=AETrainConfig(epochs=4, batch_size=64, seed=4321, verbose=False))
    assert c.best_val_loss != pytest.approx(a.best_val_loss, rel=1e-12)


def test_explicit_bottleneck_and_validation_guards():
    X_train, X_val = make_benign(120, n_features=16, seed=9), make_benign(40, n_features=16, seed=10)

    result = train_fc_dae(
        X_train, X_val, config=AETrainConfig(epochs=2, batch_size=32, bottleneck_dim=5, seed=42, verbose=False)
    )
    assert result.bottleneck_dim == 5

    with pytest.raises(ValueError, match="bottleneck_dim inválido"):
        train_fc_dae(
            X_train,
            X_val,
            config=AETrainConfig(epochs=1, bottleneck_dim=999, seed=42, verbose=False),
        )

    with pytest.raises(ValueError, match="X_val está vazio"):
        train_fc_dae(X_train, X_val[:0], config=AETrainConfig(epochs=1, verbose=False))

    with pytest.raises(ValueError, match="X_train está vazio"):
        train_fc_dae(X_train[:0], X_val, config=AETrainConfig(epochs=1, verbose=False))

    with pytest.raises(ValueError, match="features divergente"):
        train_fc_dae(
            X_train, make_benign(40, n_features=8, seed=11), config=AETrainConfig(epochs=1, verbose=False)
        )


def test_save_training_history_is_json_serializable(tmp_path):
    X_train, X_val = make_benign(150, seed=12), make_benign(60, seed=13)
    result = train_fc_dae(X_train, X_val, config=AETrainConfig(epochs=3, batch_size=64, seed=42, verbose=False))

    path = save_training_history(result, str(tmp_path / "history.json"))
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    assert len(payload["history"]) == 3
    assert payload["config"]["epochs"] == 3
    assert {"epoch", "train_loss", "val_loss", "lr"} <= set(payload["history"][0])


# --------------------------------------------------------------------------
# Extração latente (Módulo 3)
# --------------------------------------------------------------------------
def test_latent_extraction_shapes_and_batch_invariance():
    X = make_benign(257, n_features=20, seed=14)
    model = FCDAE(input_dim=20, bottleneck_dim=5)
    model.eval()

    Z_small, mse_small = extract_latent_and_mse(model, X, batch_size=16)
    Z_large, mse_large = extract_latent_and_mse(model, X, batch_size=1024)

    assert Z_small.shape == (257, 5)
    assert mse_small.shape == (257,)
    assert Z_small.dtype == np.float32 and mse_small.dtype == np.float64
    # O resultado não pode depender do tamanho do bloco de inferência.
    np.testing.assert_allclose(Z_small, Z_large, atol=1e-7)
    np.testing.assert_allclose(mse_small, mse_large, atol=1e-9)


def test_latent_mse_matches_manual_reconstruction_error():
    X = make_benign(64, n_features=12, seed=15)
    model = FCDAE(input_dim=12, bottleneck_dim=3)
    model.eval()

    _, mse = extract_latent_and_mse(model, X, batch_size=32)
    with torch.no_grad():
        reconstruction, _ = model(torch.from_numpy(X))
    expected = torch.mean((reconstruction - torch.from_numpy(X)) ** 2, dim=1).numpy()

    np.testing.assert_allclose(mse, expected, atol=1e-6)


def test_latent_frame_layout():
    Z = np.zeros((5, 3), dtype=np.float32)
    frame = latent_frame(Z, np.arange(5.0), classes=["Normal"] * 5, label_column="xAttack")
    assert list(frame.columns) == ["z_0", "z_1", "z_2", "mse", "xAttack"]
    assert len(frame) == 5


def test_latent_extraction_rejects_non_matrix():
    model = FCDAE(input_dim=4, bottleneck_dim=2)
    with pytest.raises(ValueError, match="matriz 2D"):
        extract_latent_and_mse(model, np.zeros(4, dtype=np.float32))


# --------------------------------------------------------------------------
# Calibração de limiares
# --------------------------------------------------------------------------
def test_mu_sigma_and_percentile_thresholds():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mu_sigma_threshold(values, k=0.0) == pytest.approx(3.0)
    assert mu_sigma_threshold(values, k=1.0) == pytest.approx(3.0 + values.std())
    assert percentile_threshold(values, 100.0) == pytest.approx(5.0)
    assert percentile_threshold(values, 50.0) == pytest.approx(3.0)


def test_calibrate_ae_thresholds_defaults_to_percentile_rule():
    """O padrão segue a Eq. 4 / Seção 4.7.1: tau_AE = percentil p, com p = 95."""
    rng = np.random.RandomState(0)
    mse_normal = rng.uniform(0.0, 0.01, size=5000)

    result = calibrate_ae_thresholds(mse_normal)
    assert result["tau_ae_rule"] == "percentile"
    assert result["percentile"] == 95.0
    assert result["tau_ae_ref"] == result["tau_ae_percentile"]
    # Por construção, o percentil p rejeita (100 - p)% do benigno de validação.
    assert result["expected_fpr_percentile"] == pytest.approx(0.05, abs=0.005)


def test_calibrate_ae_thresholds_honours_mu_sigma_rule():
    rng = np.random.RandomState(0)
    mse_normal = rng.uniform(0.0, 0.01, size=5000)

    result = calibrate_ae_thresholds(mse_normal, k_sigma=3.0, percentile=95.0, rule="mu_sigma")
    assert result["tau_ae_rule"] == "mu_sigma"
    assert result["tau_ae_ref"] == result["tau_ae_mu_sigma"]
    # Ambos os critérios seguem reportados, independente do escolhido.
    assert result["tau_ae_percentile"] != result["tau_ae_mu_sigma"]


def test_mu_sigma_and_p95_are_not_interchangeable():
    """A inconsistência concreta: o TCC especifica p=95, mu+3sigma opera em outro ponto.

    Sobre um MSE realista (log-normal, cauda longa à direita), o percentil 95 rejeita
    5% do benigno por construção, enquanto mu+3sigma cai muito mais à direita — uma
    diferença de quase uma ordem de magnitude no FPR operacional. Não são a mesma
    calibração, e declarar uma no texto e a outra no código muda o ponto de operação.
    """
    rng = np.random.RandomState(1)
    mse_normal = rng.lognormal(mean=-6.0, sigma=1.2, size=20000)

    result = calibrate_ae_thresholds(mse_normal, k_sigma=3.0, percentile=95.0)

    assert result["expected_fpr_percentile"] == pytest.approx(0.05, abs=0.005)
    assert result["expected_fpr_mu_sigma"] < 0.02
    # O limiar mu+3sigma fica bem acima do percentil 95 (aqui, ~1.9x).
    assert result["tau_ae_mu_sigma"] > 1.5 * result["tau_ae_percentile"]
    # E o percentil empírico em que ele cai é reportado para auditoria.
    assert result["mu_sigma_empirical_percentile"] > 97.0


def test_calibrate_ae_thresholds_rejects_unknown_rule():
    with pytest.raises(ValueError, match="rule deve ser um de"):
        calibrate_ae_thresholds([0.1, 0.2, 0.3], rule="tres_sigmas")


def test_calibration_rejects_empty_and_invalid_input():
    with pytest.raises(ValueError, match="Vetor vazio"):
        mu_sigma_threshold([])
    with pytest.raises(ValueError, match="Vetor vazio"):
        percentile_threshold([])
    with pytest.raises(ValueError, match="Percentil"):
        percentile_threshold([1.0, 2.0], percentile=101.0)

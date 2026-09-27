import numpy as np
import pytest
from src.evaluation.hybrid import HybridFusionDetector


def test_alpha_adaptivity_convergence():
    """Valida que a calibração do alfa é verdadeiramente adaptativa."""
    np.random.seed(42)
    n_val = 200
    # 100 normais (0), 100 ataques (1)
    y_val = np.array([0] * 100 + [1] * 100)

    # Cenário A: DIF discrimina perfeitamente, MSE é puro ruído
    dif_signal = np.array([0.1] * 100 + [0.9] * 100)
    mse_noise = np.random.uniform(0.001, 0.005, size=n_val)

    detector_a = HybridFusionDetector()
    detector_a.fit_scaling(mse_noise[:100])
    detector_a.calibrate(scores_dif_val=dif_signal, mse_val=mse_noise, y_val=y_val)

    # Deve convergir para alfa próximo de 1.0 (privilegiando o DIF)
    assert detector_a.alpha_ >= 0.80, f"Esperava alpha >= 0.80 quando DIF domina, obtido: {detector_a.alpha_}"

    # Cenário B: MSE discrimina perfeitamente, DIF é puro ruído
    dif_noise = np.random.uniform(0.2, 0.4, size=n_val)
    mse_signal = np.array([0.0001] * 100 + [0.05] * 100)

    detector_b = HybridFusionDetector()
    detector_b.fit_scaling(mse_signal[:100])
    detector_b.calibrate(scores_dif_val=dif_noise, mse_val=mse_signal, y_val=y_val)

    # Deve convergir para alfa próximo de 0.0 (privilegiando o MSE)
    assert detector_b.alpha_ <= 0.20, f"Esperava alpha <= 0.20 quando MSE domina, obtido: {detector_b.alpha_}"


def test_boundedness_and_anti_leakage():
    """Verifica que escores permanecem estritamente em [0, 1] mesmo com outliers extremos de teste."""
    np.random.seed(42)
    # Validação com MSE normal entre 0.001 e 0.003
    mse_val_norm = np.random.uniform(0.001, 0.003, size=200)
    y_val = np.array([0] * 150 + [1] * 50)
    scores_dif_val = np.random.uniform(0.1, 0.9, size=200)
    mse_val = np.random.uniform(0.001, 0.010, size=200)

    detector = HybridFusionDetector()
    detector.fit_scaling(mse_val_norm)
    detector.calibrate(scores_dif_val=scores_dif_val, mse_val=mse_val, y_val=y_val)

    # Teste com outliers extremos (MSE 100x maior que o visto na validação) e valores negativos
    mse_test = np.array([-0.5, 0.0, 0.002, 0.50, 10.0])
    dif_test = np.array([0.0, 0.3, 0.5, 0.9, 1.0])

    fused_scores = detector.predict_proba(scores_dif=dif_test, mse=mse_test)

    # Boundedness [0, 1] estrito
    assert np.all(fused_scores >= 0.0)
    assert np.all(fused_scores <= 1.0)
    # Outliers extremos devem saturar intencionalmente em 1.0
    assert fused_scores[-1] == 1.0


def test_or_rule_veto():
    """Verifica o comportamento da regra OR (B6b) como detector de veto especializado."""
    detector = HybridFusionDetector()
    detector.tau_dif_opt_ = 0.60
    detector.tau_ae_opt_ = 0.005

    # Caso 1: Apenas DIF detecta (score 0.70 > 0.60, mse baixo 0.001)
    pred1 = detector.predict(scores_dif=np.array([0.70]), mse=np.array([0.001]), mode="or")
    assert pred1[0] == 1

    # Caso 2: Apenas AE detecta (score baixo 0.30, mse alto 0.010 > 0.005)
    pred2 = detector.predict(scores_dif=np.array([0.30]), mse=np.array([0.010]), mode="or")
    assert pred2[0] == 1

    # Caso 3: Nenhum detecta (ambos abaixo dos limiares)
    pred3 = detector.predict(scores_dif=np.array([0.20]), mse=np.array([0.002]), mode="or")
    assert pred3[0] == 0

import numpy as np
import pytest
from src.models.dif import DeepIsolationForest

def test_normal_only_fit_without_contamination():
    """Verifica que o estimador ajusta estritamente sobre tráfego normal sem contaminação."""
    np.random.seed(42)
    # Amostras latentes benignas sintéticas (N=500, m=9)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(500, 9)).astype(np.float32)
    
    clf = DeepIsolationForest(n_estimators=10, max_samples=128, n_representations=2, random_state=42)
    # Não aceita nem requer parâmetros de contaminação arbitrária
    fitted = clf.fit(X_normal)
    assert fitted is clf
    assert hasattr(clf, "estimators_")
    assert len(clf.estimators_) == 10

def test_interface_shapes_and_keys():
    """Verifica a interface pública, chaves retornadas e conformidade dos escores em [0, 1]."""
    np.random.seed(42)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(300, 9)).astype(np.float32)
    X_eval = np.random.normal(loc=0.0, scale=1.0, size=(50, 9)).astype(np.float32)
    
    clf = DeepIsolationForest(n_estimators=10, max_samples=64, n_representations=2, random_state=42)
    clf.fit(X_normal)
    scores = clf.score_samples(X_eval)
    
    assert isinstance(scores, dict)
    assert "score_dif_deas" in scores
    assert "score_dif_standard" in scores
    
    deas = scores["score_dif_deas"]
    std = scores["score_dif_standard"]
    
    assert deas.shape == (50,)
    assert std.shape == (50,)
    assert np.all((deas >= 0.0) & (deas <= 1.0))
    assert np.all((std >= 0.0) & (std <= 1.0))

def test_lambda_zero_analytical_ablation():
    """Invariante matemático: quando lambda=0, score_dif_deas deve ser idêntico a score_dif_standard."""
    np.random.seed(42)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(400, 9)).astype(np.float32)
    X_eval = np.random.normal(loc=0.0, scale=2.0, size=(100, 9)).astype(np.float32)
    
    clf = DeepIsolationForest(n_estimators=20, max_samples=128, n_representations=4, lambda_deas=0.0, random_state=42)
    clf.fit(X_normal)
    scores = clf.score_samples(X_eval)
    
    # Com lambda=0, o termo de desvio desaparece e recai exatamente no Isolation Forest tradicional
    np.testing.assert_allclose(scores["score_dif_deas"], scores["score_dif_standard"], atol=1e-7)

def test_numerical_stability_on_constant_data():
    """Garante estabilidade numérica contra nós de variância nula e divisão por zero."""
    # Matriz constante com zero variância
    X_constant = np.ones((200, 9), dtype=np.float32) * 0.5
    
    clf = DeepIsolationForest(n_estimators=5, max_samples=64, n_representations=1, lambda_deas=0.5, random_state=42)
    clf.fit(X_constant)
    scores = clf.score_samples(X_constant)
    
    assert not np.isnan(scores["score_dif_deas"]).any()
    assert not np.isinf(scores["score_dif_deas"]).any()
    assert not np.isnan(scores["score_dif_standard"]).any()
    assert not np.isinf(scores["score_dif_standard"]).any()

def test_lambda_sensitivity_monotonicity():
    """Verifica se o fator lambda modula a sensibilidade do desvio contínuo."""
    np.random.seed(42)
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(300, 9)).astype(np.float32)
    X_outliers = np.random.uniform(low=5.0, high=10.0, size=(30, 9)).astype(np.float32)
    
    clf_low = DeepIsolationForest(n_estimators=20, max_samples=128, lambda_deas=0.1, random_state=42)
    clf_low.fit(X_normal)
    scores_low = clf_low.score_samples(X_outliers)["score_dif_deas"]
    
    clf_high = DeepIsolationForest(n_estimators=20, max_samples=128, lambda_deas=0.9, random_state=42)
    clf_high.fit(X_normal)
    scores_high = clf_high.score_samples(X_outliers)["score_dif_deas"]
    
    # Ambas as pontuações devem ser válidas e diferentes devido à ponderação distinta
    assert not np.allclose(scores_low, scores_high)

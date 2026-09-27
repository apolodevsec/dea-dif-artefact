import numpy as np
import pytest
from src.models.dif import DeepIsolationForest


def test_ghost_regions_mitigation():
    """Validação da hipótese central: mitigação do problema de ghost regions pelo DEAS.
    
    Cenário: Dois clusters densos em x = -5 e x = +5.
    A região entre -1 e +1 é vazia (ghost region).
    O iForest clássico frequentemente atribui comprimentos de caminho longos (baixo score de anomalia)
    a pontos nessa região vazia devido a cortes axiais grosseiros.
    O DEAS penaliza essa região vazia calculando o desvio relativo contínuo em relação aos nós,
    atribuindo scores de anomalia superiores aos do iForest padrão.
    """
    np.random.seed(42)
    # Cluster 1: centrado em -5.0
    c1 = np.random.normal(loc=-5.0, scale=0.5, size=(400, 9)).astype(np.float32)
    # Cluster 2: centrado em +5.0
    c2 = np.random.normal(loc=5.0, scale=0.5, size=(400, 9)).astype(np.float32)
    X_train = np.vstack([c1, c2])

    # Pontos de teste:
    # 1. Pontos legítimos densos nos clusters
    X_dense = np.vstack([
        np.random.normal(loc=-5.0, scale=0.3, size=(50, 9)),
        np.random.normal(loc=5.0, scale=0.3, size=(50, 9)),
    ]).astype(np.float32)

    # 2. Pontos na ghost region (vazia, entre os clusters: ~0.0)
    X_ghost = np.random.normal(loc=0.0, scale=0.3, size=(50, 9)).astype(np.float32)

    # 3. Outliers extremos longe de tudo: ~15.0
    X_outliers = np.random.normal(loc=15.0, scale=0.5, size=(50, 9)).astype(np.float32)

    clf = DeepIsolationForest(
        n_estimators=100,
        max_samples=256,
        n_representations=10,
        lambda_deas=0.5,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_train)

    scores_dense = clf.score_samples(X_dense)
    scores_ghost = clf.score_samples(X_ghost)
    scores_outliers = clf.score_samples(X_outliers)

    # 1. Discriminação de outliers: Outliers extremos devem ter score significativamente maior que pontos densos
    assert np.mean(scores_outliers["score_dif_deas"]) > np.mean(scores_dense["score_dif_deas"]) + 0.15

    # 2. Mitigação de Ghost Regions:
    # Na região vazia, o DEAS deve acusar maior anomalia que o score padrão
    mean_ghost_deas = np.mean(scores_ghost["score_dif_deas"])
    mean_ghost_std = np.mean(scores_ghost["score_dif_standard"])
    assert mean_ghost_deas > mean_ghost_std, (
        f"Esperava que score DEAS ({mean_ghost_deas:.4f}) fosse superior ao standard ({mean_ghost_std:.4f}) na ghost region"
    )

    # 3. Preservação de normalidade: Amostras densas devem ter escores bem contidos
    assert np.mean(scores_dense["score_dif_deas"]) < 0.55


def test_batch_invariance_and_determinism():
    """Verifica invariância por tamanho de mini-batch e determinismo com semente fixada."""
    np.random.seed(123)
    X = np.random.normal(size=(500, 9)).astype(np.float32)

    clf1 = DeepIsolationForest(n_estimators=20, max_samples=64, n_representations=2, random_state=999, n_jobs=1)
    clf1.fit(X)

    # Inferência com batch_size=100 vs batch_size=500
    res_b100 = clf1.score_samples(X, batch_size=100)
    res_b500 = clf1.score_samples(X, batch_size=500)

    np.testing.assert_allclose(res_b100["score_dif_deas"], res_b500["score_dif_deas"], atol=1e-7)
    np.testing.assert_allclose(res_b100["score_dif_standard"], res_b500["score_dif_standard"], atol=1e-7)

    # Determinismo: novo modelo com mesma seed e mesmo fit deve gerar resultados idênticos
    clf2 = DeepIsolationForest(n_estimators=20, max_samples=64, n_representations=2, random_state=999, n_jobs=1)
    clf2.fit(X)
    res_clf2 = clf2.score_samples(X)

    np.testing.assert_allclose(res_b100["score_dif_deas"], res_clf2["score_dif_deas"], atol=1e-7)
    np.testing.assert_allclose(res_b100["score_dif_standard"], res_clf2["score_dif_standard"], atol=1e-7)


def test_representation_decoupling_strict_mode():
    """Verifica que o modo estrito r = t = 10 constrói 1 rede por árvore de isolamento."""
    np.random.seed(42)
    X = np.random.normal(size=(100, 9)).astype(np.float32)

    clf_strict = DeepIsolationForest(n_estimators=10, max_samples=64, n_representations=10, random_state=42, n_jobs=1)
    clf_strict.fit(X)

    assert len(clf_strict.networks_) == 10
    # Cada árvore aponta para um net_idx distinto de 0 a 9
    net_indices = [net_idx for net_idx, _ in clf_strict.estimators_]
    assert net_indices == list(range(10))

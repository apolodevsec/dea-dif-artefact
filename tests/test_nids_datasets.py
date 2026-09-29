import numpy as np
import pandas as pd
import pytest

from src.data.nids_datasets import (
    dataset_display_name,
    detect_label_column,
    detect_normal_label,
    load_partitions,
    normalize_class_name,
    read_label_column,
    read_parquet_float32,
    resolve_dataset_dirs,
)


# --------------------------------------------------------------------------
# Fixture: replica o layout real dos artefatos processados
# --------------------------------------------------------------------------
def write_dataset(
    directory,
    label_column: str,
    normal_label: str,
    attack_counts: dict,
    n_normal_train: int = 400,
    n_normal_test: int = 200,
    n_features: int = 16,
    seed: int = 7,
):
    """Materializa um dataset sintético no layout `*-processados`."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    def normal_block(n):
        return rng.uniform(0.0, 0.35, size=(n, n_features)).astype(np.float64)

    def attack_block(n, offset):
        return np.clip(rng.normal(0.5 + offset, 0.12, size=(n, n_features)), 0.0, 1.0)

    cols = [f"f{i}" for i in range(n_features)]

    # --- Treino: benigno + ataques ---
    X_normal_train = normal_block(n_normal_train)
    blocks = [X_normal_train]
    labels = [normal_label] * n_normal_train
    for offset, (family, count) in enumerate(attack_counts.items()):
        blocks.append(attack_block(count, 0.1 * offset))
        labels.extend([family] * count)

    X_train = np.vstack(blocks)
    pd.DataFrame(X_train, columns=cols).to_parquet(directory / "X_train_baseline.parquet", index=False)
    pd.DataFrame({label_column: labels}).to_parquet(directory / "y_train_baseline.parquet", index=False)
    # Por contrato do artefato, este arquivo é o subconjunto benigno do treino.
    pd.DataFrame(X_normal_train, columns=cols).to_parquet(
        directory / "X_train_autoencoder.parquet", index=False
    )

    # --- Teste ---
    test_blocks = [normal_block(n_normal_test)]
    test_labels = [normal_label] * n_normal_test
    for offset, (family, count) in enumerate(attack_counts.items()):
        n_test_family = max(2, count // 2)
        test_blocks.append(attack_block(n_test_family, 0.1 * offset))
        test_labels.extend([family] * n_test_family)

    X_test = np.vstack(test_blocks)
    pd.DataFrame(X_test, columns=cols).to_parquet(directory / "X_test.parquet", index=False)
    pd.DataFrame({label_column: test_labels}).to_parquet(directory / "y_test.parquet", index=False)
    return directory


@pytest.fixture()
def nsl_like(tmp_path):
    return write_dataset(
        tmp_path / "NSL-KDD-processados",
        label_column="xAttack",
        normal_label="Normal",
        attack_counts={"DoS": 150, "Probe": 60, "R2L": 20, "U2R": 3},
    )


# --------------------------------------------------------------------------
# Detecção de esquema e normalização de rótulos
# --------------------------------------------------------------------------
def test_detect_label_column_per_dataset():
    assert detect_label_column(["duration", "xAttack"]) == "xAttack"
    assert detect_label_column(["attack_cat"]) == "attack_cat"
    assert detect_label_column(["Flow Duration", "Label"]) == "Label"


def test_detect_label_column_rejects_ambiguous_schema():
    with pytest.raises(ValueError, match="coluna de rótulo"):
        detect_label_column(["foo", "bar"])


def test_detect_normal_label_per_dataset():
    assert detect_normal_label(["Normal", "DoS", "Probe"]) == "Normal"
    assert detect_normal_label(["BENIGN", "DDoS", "PortScan"]) == "BENIGN"
    with pytest.raises(ValueError, match="Nenhum rótulo benigno"):
        detect_normal_label(["DoS", "Probe"])


def test_normalize_class_name_repairs_cicids_mojibake():
    # O CICIDS2017 traz U+FFFD no lugar do separador em "Web Attack – XSS".
    assert normalize_class_name("Web Attack � XSS") == "Web Attack - XSS"
    assert normalize_class_name("  DoS   Hulk ") == "DoS Hulk"
    # Hifens legítimos são preservados.
    assert normalize_class_name("SSH-Patator") == "SSH-Patator"


def test_read_parquet_float32_and_labels(nsl_like):
    X, columns = read_parquet_float32(str(nsl_like / "X_test.parquet"))
    assert X.dtype == np.float32
    assert len(columns) == X.shape[1]
    assert np.isfinite(X).all()

    classes, column = read_label_column(str(nsl_like / "y_test.parquet"))
    assert column == "xAttack"
    assert len(classes) == len(X)

    # Leitura por subconjunto e ordem explícita de colunas
    subset, names = read_parquet_float32(str(nsl_like / "X_test.parquet"), columns=columns[:3], max_rows=10)
    assert subset.shape == (10, 3)
    assert names == columns[:3]
    np.testing.assert_allclose(subset, X[:10, :3])


def test_read_parquet_float32_rejects_unknown_column(nsl_like):
    with pytest.raises(ValueError, match="Colunas ausentes"):
        read_parquet_float32(str(nsl_like / "X_test.parquet"), columns=["coluna_inexistente"])


def test_dataset_display_name_strips_suffix():
    assert dataset_display_name("/data/CICIDS2017-processados") == "CICIDS2017"
    assert dataset_display_name("/data/UNSW-NB15_processados") == "UNSW-NB15"
    assert dataset_display_name("/data/meu-dataset") == "meu-dataset"


def test_resolve_dataset_dirs_discovers_and_filters(tmp_path):
    write_dataset(tmp_path / "NSL-KDD-processados", "xAttack", "Normal", {"DoS": 40})
    write_dataset(tmp_path / "CICIDS2017-processados", "Label", "BENIGN", {"DDoS": 40})
    (tmp_path / "pasta-vazia").mkdir()

    todos = resolve_dataset_dirs(str(tmp_path))
    assert len(todos) == 2

    filtrado = resolve_dataset_dirs(str(tmp_path), ["CICIDS2017"])
    assert len(filtrado) == 1
    assert "CICIDS2017" in filtrado[0]

    with pytest.raises(FileNotFoundError, match="não encontrado"):
        resolve_dataset_dirs(str(tmp_path), ["UNSW-NB15"])


# --------------------------------------------------------------------------
# Particionamento leakage-free
# --------------------------------------------------------------------------
def test_partitions_are_benign_only_where_required(nsl_like):
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)

    assert parts.label_column == "xAttack"
    assert parts.normal_label == "Normal"
    assert parts.dataset_name == "NSL-KDD"

    # O FC-DAE e o DIF só veem tráfego benigno.
    assert len(parts.X_ae_train) > 0
    assert len(parts.X_ae_val) > 0
    # A validação rotulada contém benigno e ataques.
    assert parts.y_val.min() == 0 and parts.y_val.max() == 1
    assert parts.val_normal_mask.sum() == len(parts.X_ae_val)
    # O bloco benigno da validação é exatamente X_ae_val (prefixo), o que permite
    # derivar o MSE benigno sem uma passagem extra pelo autoencoder.
    np.testing.assert_allclose(parts.X_val[: len(parts.X_ae_val)], parts.X_ae_val)


def test_benign_partitions_are_disjoint(nsl_like):
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)

    train_rows = {row.tobytes() for row in parts.X_ae_train}
    val_rows = {row.tobytes() for row in parts.X_ae_val}
    assert train_rows.isdisjoint(val_rows)
    assert len(train_rows) + len(val_rows) == len(parts.X_ae_train) + len(parts.X_ae_val)


def test_validation_attacks_come_from_train_not_test(nsl_like):
    """Invariante central anti-vazamento: o teste nunca alimenta a calibração."""
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)

    test_rows = {row.tobytes() for row in parts.X_test}
    for name, matrix in [
        ("X_ae_train", parts.X_ae_train),
        ("X_ae_val", parts.X_ae_val),
        ("X_val", parts.X_val),
        ("X_sup_attacks", parts.X_sup_attacks),
    ]:
        overlap = sum(1 for row in matrix if row.tobytes() in test_rows)
        assert overlap == 0, f"{name} compartilha {overlap} linhas com o conjunto de teste cego"


def test_attack_split_is_stratified_by_family(nsl_like):
    parts = load_partitions(str(nsl_like), val_attack_ratio=0.2, random_state=42, verbose=False)
    split = parts.meta["attack_split_train"]

    assert set(split) == {"DoS", "Probe", "R2L", "U2R"}
    for family, counts in split.items():
        assert counts["val"] + counts["sup_train"] == counts["total"]
        # Toda família observável na validação, inclusive as raras.
        assert counts["val"] >= 1, family
        assert counts["sup_train"] >= 1, family

    # A proporção de ~20% é respeitada nas famílias suficientemente populosas.
    assert split["DoS"]["val"] == pytest.approx(0.2 * split["DoS"]["total"], abs=1)


def test_rare_classes_flagged_from_test_set(nsl_like):
    parts = load_partitions(str(nsl_like), rare_threshold=15, random_state=42, verbose=False)
    # U2R tem 3 instâncias no treino e 2 no teste -> rara.
    assert "U2R" in parts.rare_classes
    assert parts.rare_classes["U2R"] < 15
    assert "DoS" not in parts.rare_classes


def test_autoencoder_file_consistency_is_verified(nsl_like):
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)
    check = parts.meta["autoencoder_file_check"]
    assert check["present"] is True
    assert check["row_count_matches"] is True
    assert check["rows"] == check["benign_rows_in_baseline"]


def test_partition_is_deterministic_for_a_fixed_seed(nsl_like):
    a = load_partitions(str(nsl_like), random_state=123, verbose=False)
    b = load_partitions(str(nsl_like), random_state=123, verbose=False)
    np.testing.assert_allclose(a.X_ae_train, b.X_ae_train)
    np.testing.assert_array_equal(a.y_val, b.y_val)

    c = load_partitions(str(nsl_like), random_state=999, verbose=False)
    assert not np.allclose(a.X_ae_train, c.X_ae_train)


def test_stratified_subsampling_respects_caps(nsl_like):
    parts = load_partitions(
        str(nsl_like), max_train_rows=200, max_test_rows=100, random_state=42, verbose=False
    )
    assert len(parts.X_ae_train) + len(parts.X_ae_val) + len(parts.X_sup_attacks) + parts.sizes()[
        "val_attacks"
    ] <= 210
    assert len(parts.X_test) <= 110
    # A subamostragem preserva a presença de todas as classes do teste.
    assert len(np.unique(parts.classes_test)) == 5


def test_invalid_ratios_are_rejected(nsl_like):
    with pytest.raises(ValueError, match="val_normal_ratio"):
        load_partitions(str(nsl_like), val_normal_ratio=0.0, verbose=False)
    with pytest.raises(ValueError, match="val_attack_ratio"):
        load_partitions(str(nsl_like), val_attack_ratio=1.0, verbose=False)


def _replicate_counts(counts: dict) -> np.ndarray:
    return np.array([fam for fam, n in counts.items() for _ in range(n)], dtype=object)


@pytest.mark.parametrize(
    "dataset, train_counts, test_counts",
    [
        # Contagens reais medidas nos artefatos processados fornecidos.
        (
            "NSL-KDD",
            {"Normal": 53876, "DoS": 37075, "Probe": 9751, "R2L": 2624, "U2R": 178},
            {"Normal": 23090, "DoS": 15890, "Probe": 4179, "R2L": 1125, "U2R": 76},
        ),
        (
            "UNSW-NB15",
            {
                "Normal": 60005, "Exploits": 18169, "Fuzzers": 13845, "Reconnaissance": 6625,
                "Generic": 4907, "DoS": 3691, "Analysis": 1356, "Backdoor": 1264,
                "Shellcode": 963, "Worms": 113,
            },
            {
                "Normal": 25717, "Exploits": 7787, "Fuzzers": 5933, "Reconnaissance": 2839,
                "Generic": 2103, "DoS": 1582, "Analysis": 581, "Backdoor": 541,
                "Shellcode": 413, "Worms": 49,
            },
        ),
    ],
)
def test_real_artifacts_are_stratified_resplits_not_canonical(dataset, train_counts, test_counts):
    """Regressão factual: os splits fornecidos replicam as proporções do treino.

    Nenhuma família é exclusiva do teste e a maior divergência de proporção fica na
    quarta casa decimal — assinatura de um refatiamento aleatório estratificado 70/30.
    Em consequência, o cenário *zero-day* do split canônico (KDDTest+ no NSL-KDD) NÃO
    está sendo avaliado.
    """
    from src.data.nids_datasets import diagnose_split_provenance

    provenance = diagnose_split_provenance(
        _replicate_counts(train_counts), _replicate_counts(test_counts)
    )

    assert provenance["looks_like_stratified_resplit"] is True, dataset
    assert provenance["preserves_novel_attack_scenario"] is False, dataset
    assert provenance["families_only_in_test"] == [], dataset
    assert provenance["max_class_proportion_delta_pp"] < 0.01, dataset
    assert provenance["test_fraction"] == pytest.approx(0.30, abs=0.0005), dataset
    assert "NÃO está preservado" in provenance["verdict"]


def test_split_provenance_is_attached_to_partitions(nsl_like):
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)
    provenance = parts.meta["split_provenance"]

    assert set(provenance["class_share_percent"]) == {"train", "test"}
    assert provenance["n_train"] + provenance["n_test"] == provenance["n_total"]
    assert "verdict" in provenance


def test_split_provenance_detects_novel_attack_scenario(tmp_path):
    """Um split canônico traz famílias ausentes do treino e distribuição deslocada."""
    from src.data.nids_datasets import diagnose_split_provenance

    train = np.array(["Normal"] * 700 + ["DoS"] * 300, dtype=object)
    # Teste com proporção deslocada e uma família inédita (novidade/zero-day).
    test = np.array(["Normal"] * 100 + ["DoS"] * 80 + ["Worms"] * 20, dtype=object)

    provenance = diagnose_split_provenance(train, test)
    assert provenance["looks_like_stratified_resplit"] is False
    assert provenance["preserves_novel_attack_scenario"] is True
    assert provenance["families_only_in_test"] == ["Worms"]
    assert provenance["max_class_proportion_delta_pp"] > 1.0


def test_temporal_support_reports_b5_as_blocked(nsl_like):
    """Sem coluna Timestamp nem arquivo paralelo, B5 não é executável."""
    parts = load_partitions(str(nsl_like), random_state=42, verbose=False)
    temporal = parts.meta["temporal_support"]

    assert temporal["temporal_columns_in_features"] == []
    assert temporal["sidecar_files_present"] == []
    assert temporal["b5_lstm_executable"] is False


def test_temporal_support_detects_timestamp_column(tmp_path):
    """Se o Módulo 1 preservar a coluna temporal, B5 passa a ser viável."""
    from src.data.nids_datasets import detect_temporal_columns

    assert detect_temporal_columns(["Flow Duration", "Timestamp"]) == ["Timestamp"]
    assert detect_temporal_columns(["dur", "spkts"]) == []

    directory = write_dataset(tmp_path / "TEMPORAL-processados", "Label", "BENIGN", {"DDoS": 40})
    # Simula o artefato paralelo previsto em eval.md 4.8.5.
    (directory / "timestamps.parquet").write_bytes(b"")
    parts = load_partitions(str(directory), random_state=42, verbose=False)
    assert parts.meta["temporal_support"]["sidecar_files_present"] == ["timestamps.parquet"]
    assert parts.meta["temporal_support"]["b5_lstm_executable"] is True


def test_global_partition_share_is_reported(nsl_like):
    """As frações efetivas diferem do 70/15/15 nominal e precisam ser explícitas."""
    parts = load_partitions(str(nsl_like), val_normal_ratio=0.15, random_state=42, verbose=False)
    share = parts.meta["global_partition_share"]

    assert share["normal"]["train_unsupervised"] + share["normal"]["validation"] + share["normal"][
        "test"
    ] == pytest.approx(1.0)
    assert share["attack"]["validation"] + share["attack"]["reserved_supervised"] + share["attack"][
        "test"
    ] == pytest.approx(1.0)
    # O teste do artefato carrega ~1/3 do benigno, logo o treino não chega a 70%.
    assert share["normal"]["train_unsupervised"] < 0.70


def test_cicids_like_labels_are_normalized_end_to_end(tmp_path):
    directory = write_dataset(
        tmp_path / "CICIDS2017-processados",
        label_column="Label",
        normal_label="BENIGN",
        attack_counts={"DoS Hulk": 120, "Web Attack � XSS": 40, "Heartbleed": 4},
    )
    parts = load_partitions(str(directory), random_state=42, verbose=False)

    assert parts.normal_label == "BENIGN"
    families = set(parts.classes_test)
    assert "Web Attack - XSS" in families
    assert not any("�" in f for f in families)

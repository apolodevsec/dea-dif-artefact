"""Carregamento e particionamento *leakage-free* dos datasets NIDS pré-processados.

Layout esperado em cada diretório de dataset (um diretório por dataset):

```text
<dataset_dir>/
├── X_train_autoencoder.parquet   # tráfego estritamente benigno (Módulo 1)
├── X_train_baseline.parquet      # treino completo (benigno + ataques)
├── y_train_baseline.parquet      # rótulos multiclasse do treino
├── X_test.parquet                # conjunto de teste cego
├── y_test.parquet                # rótulos multiclasse do teste
└── preprocessor.joblib           # opcional (não requerido no treino)
```

A coluna de rótulo varia por dataset (`xAttack` no NSL-KDD, `attack_cat` no
UNSW-NB15, `Label` no CICIDS2017) e é detectada automaticamente, assim como o
valor que identifica o tráfego benigno (`Normal` / `BENIGN`).

O protocolo de particionamento implementado aqui preserva o conjunto de teste
fornecido como **cego**: toda a calibração (limiares, alfa da fusão, ponto de
operação de Youden) consome exclusivamente partições derivadas de
`X_train_baseline`.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Nomes de coluna de rótulo conhecidos, em ordem de precedência.
LABEL_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "xAttack",
    "attack_cat",
    "Label",
    "label",
    "class",
    "Class",
)

# Valores que identificam tráfego benigno, em ordem de precedência.
NORMAL_LABEL_CANDIDATES: Tuple[str, ...] = (
    "BENIGN",
    "Normal",
    "NORMAL",
    "benign",
    "normal",
    "0",
)

# Rótulos considerados ausentes/inválidos e descartados com aviso.
MISSING_LABEL_TOKENS = frozenset({"", "nan", "none", "null", "na", "<na>"})

_REQUIRED_FILES = ("X_train_baseline.parquet", "y_train_baseline.parquet", "X_test.parquet", "y_test.parquet")

_PARQUET_READ_BATCH = 131_072


def normalize_class_name(raw: Any) -> str:
    """Normaliza o nome de uma família de ataque para uso em tabelas e relatórios.

    Colapsa espaços redundantes e substitui o caractere de substituição Unicode
    (U+FFFD) presente nos rótulos do CICIDS2017 (`"Web Attack � XSS"`) por um
    hífen, preservando hifens legítimos como em `"SSH-Patator"`.
    """
    text = str(raw).strip().replace("�", "-")
    return re.sub(r"\s+", " ", text)


def detect_label_column(columns: Sequence[str]) -> str:
    """Identifica a coluna de rótulo entre os nomes conhecidos."""
    for candidate in LABEL_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    if len(columns) == 1:
        return str(columns[0])
    raise ValueError(
        f"Não foi possível detectar a coluna de rótulo entre {list(columns)}. "
        f"Candidatos conhecidos: {list(LABEL_COLUMN_CANDIDATES)}."
    )


def detect_normal_label(classes: Sequence[str]) -> str:
    """Identifica o valor que representa o tráfego benigno no vetor de rótulos."""
    unique = set(map(str, classes))
    matches = [c for c in NORMAL_LABEL_CANDIDATES if c in unique]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Múltiplos rótulos benignos candidatos encontrados: {matches}. "
            f"Informe explicitamente o valor correto via 'normal_label'."
        )
    raise ValueError(
        f"Nenhum rótulo benigno reconhecido entre {sorted(unique)}. "
        f"Candidatos conhecidos: {list(NORMAL_LABEL_CANDIDATES)}."
    )


def read_parquet_float32(
    path: str,
    columns: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Lê um parquet numérico direto para uma matriz `float32` pré-alocada.

    A leitura é feita por *record batches* para evitar o pico de memória da
    materialização intermediária em `float64` do pandas — relevante para o
    CICIDS2017, cujo treino possui 1,56 milhão de fluxos.

    Valores não finitos (NaN/±Inf) são substituídos por 0.0 com aviso explícito.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo parquet não encontrado: '{os.path.abspath(path)}'")

    parquet_file = pq.ParquetFile(path)
    names = [str(c) for c in (columns if columns is not None else parquet_file.schema_arrow.names)]

    missing = [n for n in names if n not in set(parquet_file.schema_arrow.names)]
    if missing:
        raise ValueError(f"Colunas ausentes em '{path}': {missing}")

    total_rows = parquet_file.metadata.num_rows
    if max_rows is not None:
        total_rows = min(int(max_rows), total_rows)

    out = np.empty((total_rows, len(names)), dtype=np.float32)
    written = 0

    for batch in parquet_file.iter_batches(batch_size=_PARQUET_READ_BATCH, columns=names):
        if written >= total_rows:
            break
        take = min(batch.num_rows, total_rows - written)
        for j in range(len(names)):
            col = batch.column(j).to_numpy(zero_copy_only=False)
            out[written : written + take, j] = col[:take]
        written += take

    if written < total_rows:  # parquet truncado em relação aos metadados
        out = out[:written]

    non_finite = ~np.isfinite(out)
    n_non_finite = int(non_finite.sum())
    if n_non_finite:
        warnings.warn(
            f"'{os.path.basename(path)}': {n_non_finite} valores não finitos substituídos por 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        out[non_finite] = 0.0

    return out, names


def read_label_column(
    path: str,
    label_column: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """Lê a coluna de rótulo de um parquet e devolve os nomes de classe normalizados."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo parquet não encontrado: '{os.path.abspath(path)}'")

    df = pd.read_parquet(path)
    column = label_column or detect_label_column(list(df.columns))
    series = df[column]
    if max_rows is not None:
        series = series.iloc[: int(max_rows)]

    classes = np.array([normalize_class_name(v) for v in series.to_numpy()], dtype=object)
    return classes, column


def _valid_label_mask(classes: np.ndarray) -> np.ndarray:
    return np.array([str(c).strip().lower() not in MISSING_LABEL_TOKENS for c in classes], dtype=bool)


def detect_temporal_columns(columns: Sequence[str]) -> List[str]:
    """Localiza colunas de carimbo temporal entre as features.

    O baseline B5 (LSTM-AE) exige janelas cronológicas de `W` fluxos ordenadas por
    `Timestamp`. Se o pré-processamento descartou a coluna temporal e não a preservou
    num artefato paralelo, B5 não tem base para ser executado.
    """
    keywords = ("timestamp", "time", "date", "hour", "epoch", "stamp")
    return [c for c in columns if any(k in str(c).lower() for k in keywords)]


def diagnose_split_provenance(
    train_classes: np.ndarray,
    test_classes: np.ndarray,
    proportion_tolerance_pp: float = 0.05,
) -> Dict[str, Any]:
    """Infere se o split treino/teste do artefato é canônico ou um refatiamento estratificado.

    Splits canônicos de NIDS (KDDTest+ do NSL-KDD, por exemplo) deslocam
    deliberadamente a distribuição de classes e introduzem famílias de ataque
    ausentes do treino — é esse desbalanceamento que cria o cenário *zero-day*.
    Um refatiamento aleatório estratificado, ao contrário, replica as proporções do
    treino no teste e não contém nenhuma família nova.

    A assinatura é decisiva: se a maior divergência de proporção entre treino e teste
    fica abaixo de uma fração de ponto percentual **e** nenhuma família é exclusiva do
    teste, o split é um refatiamento e o cenário de generalização a ataques inéditos
    **não** está sendo avaliado.
    """
    n_train, n_test = len(train_classes), len(test_classes)
    train_families, train_counts = np.unique(train_classes, return_counts=True)
    test_families, test_counts = np.unique(test_classes, return_counts=True)

    train_share = {str(f): c / max(1, n_train) * 100.0 for f, c in zip(train_families, train_counts)}
    test_share = {str(f): c / max(1, n_test) * 100.0 for f, c in zip(test_families, test_counts)}

    all_families = sorted(set(train_share) | set(test_share))
    deltas = {f: test_share.get(f, 0.0) - train_share.get(f, 0.0) for f in all_families}
    max_delta = max((abs(d) for d in deltas.values()), default=0.0)

    test_only = sorted(set(test_share) - set(train_share))
    train_only = sorted(set(train_share) - set(test_share))

    total = n_train + n_test
    test_fraction = n_test / max(1, total)

    stratified_resplit = bool(max_delta < proportion_tolerance_pp and not test_only)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "n_total": total,
        "test_fraction": float(test_fraction),
        "max_class_proportion_delta_pp": float(max_delta),
        "families_only_in_test": test_only,
        "families_only_in_train": train_only,
        "looks_like_stratified_resplit": stratified_resplit,
        "preserves_novel_attack_scenario": bool(test_only),
        "class_share_percent": {
            "train": {k: round(v, 4) for k, v in train_share.items()},
            "test": {k: round(v, 4) for k, v in test_share.items()},
        },
        "verdict": (
            "Refatiamento aleatório estratificado: as proporções de classe do teste "
            "replicam as do treino e nenhuma família é exclusiva do teste. O split "
            "canônico do dataset (e com ele o cenário zero-day de ataques inéditos) "
            "NÃO está preservado."
            if stratified_resplit
            else "Split com distribuição deslocada e/ou famílias exclusivas do teste, "
            "compatível com o particionamento canônico do dataset."
        ),
    }


def _stratified_row_subsample(
    classes: np.ndarray,
    max_rows: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Subamostra índices preservando a proporção de cada classe (mínimo 1 por classe)."""
    n = len(classes)
    if max_rows >= n:
        return np.arange(n)

    keep: List[np.ndarray] = []
    unique = np.unique(classes)
    fraction = max_rows / n
    for cls in unique:
        idx = np.flatnonzero(classes == cls)
        quota = max(1, int(round(len(idx) * fraction)))
        quota = min(quota, len(idx))
        keep.append(rng.permutation(idx)[:quota])

    selected = np.concatenate(keep)
    selected.sort()
    return selected


@dataclass
class NidsPartitions:
    """Partições prontas para consumo pelos Módulos 2 a 5, livres de vazamento.

    - `X_ae_train` / `X_ae_val`: tráfego **exclusivamente benigno**. Treinam o
      FC-DAE e o DIF (`X_ae_train`) e calibram os limiares não-supervisionados
      `tau_AE` e `tau_DIF_ref` (`X_ae_val`).
    - `X_val` / `y_val` / `classes_val`: validação **rotulada** (benigno de
      validação + fração estratificada dos ataques de treino). Única fonte para o
      ponto de operação de Youden e para o alfa da fusão linear.
    - `X_sup_attacks`: ataques de treino não usados na validação, reservados ao
      baseline supervisionado B4.
    - `X_test` / `y_test` / `classes_test`: conjunto de teste cego, tocado uma
      única vez na avaliação final.
    """

    dataset_name: str
    label_column: str
    normal_label: str
    feature_columns: List[str]

    X_ae_train: np.ndarray
    X_ae_val: np.ndarray

    X_val: np.ndarray
    y_val: np.ndarray
    classes_val: np.ndarray
    val_normal_mask: np.ndarray

    X_sup_attacks: np.ndarray
    classes_sup_attacks: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    classes_test: np.ndarray

    rare_classes: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns)

    def sizes(self) -> Dict[str, int]:
        return {
            "ae_train_normal": int(len(self.X_ae_train)),
            "ae_val_normal": int(len(self.X_ae_val)),
            "val_total": int(len(self.X_val)),
            "val_normal": int(np.sum(self.val_normal_mask)),
            "val_attacks": int(len(self.X_val) - np.sum(self.val_normal_mask)),
            "sup_train_attacks": int(len(self.X_sup_attacks)),
            "test_total": int(len(self.X_test)),
            "test_normal": int(np.sum(self.y_test == 0)),
            "test_attacks": int(np.sum(self.y_test == 1)),
        }


def resolve_dataset_dirs(
    data_root: str,
    dataset_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Localiza os diretórios de dataset sob `data_root`.

    Um diretório é considerado um dataset quando contém os parquets obrigatórios.
    Quando `dataset_names` é informado, cada nome é casado por substring
    case-insensitive (ex.: `"NSL-KDD"` casa `"NSL-KDD-processados"`).
    """
    if not os.path.isdir(data_root):
        raise NotADirectoryError(f"Diretório raiz de dados inexistente: '{data_root}'")

    candidates: List[str] = []
    if all(os.path.exists(os.path.join(data_root, f)) for f in _REQUIRED_FILES):
        candidates.append(data_root)

    for entry in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, entry)
        if os.path.isdir(full) and all(os.path.exists(os.path.join(full, f)) for f in _REQUIRED_FILES):
            candidates.append(full)

    if not candidates:
        raise FileNotFoundError(
            f"Nenhum dataset válido encontrado sob '{data_root}'. "
            f"Cada diretório deve conter: {list(_REQUIRED_FILES)}."
        )

    if dataset_names is None:
        return candidates

    selected: List[str] = []
    for name in dataset_names:
        needle = name.strip().lower()
        hits = [c for c in candidates if needle in os.path.basename(c).lower()]
        if not hits:
            raise FileNotFoundError(
                f"Dataset '{name}' não encontrado sob '{data_root}'. "
                f"Disponíveis: {[os.path.basename(c) for c in candidates]}."
            )
        for hit in hits:
            if hit not in selected:
                selected.append(hit)
    return selected


def dataset_display_name(dataset_dir: str) -> str:
    """Deriva um nome curto de dataset a partir do nome do diretório."""
    base = os.path.basename(os.path.normpath(dataset_dir))
    for suffix in ("-processados", "_processados", "-processed", "_processed"):
        if base.lower().endswith(suffix):
            return base[: -len(suffix)]
    return base


def load_partitions(
    dataset_dir: str,
    val_normal_ratio: float = 0.15,
    val_attack_ratio: float = 0.20,
    rare_threshold: int = 15,
    random_state: int = 42,
    label_column: Optional[str] = None,
    normal_label: Optional[str] = None,
    max_train_rows: Optional[int] = None,
    max_test_rows: Optional[int] = None,
    verbose: bool = True,
) -> NidsPartitions:
    """Carrega um dataset e monta as partições do protocolo *leakage-free*.

    :param val_normal_ratio: fração do tráfego benigno de treino reservada à
        validação (o restante treina o FC-DAE e o DIF).
    :param val_attack_ratio: fração estratificada dos ataques de treino movida
        para a validação rotulada; o complemento fica disponível ao baseline
        supervisionado B4.
    :param rare_threshold: limite abaixo do qual uma família de ataque do teste é
        marcada como estatisticamente rara na ressalva do relatório.
    :param max_train_rows: subamostragem estratificada opcional do treino.
    :param max_test_rows: subamostragem estratificada opcional do teste.
    """
    if not 0.0 < val_normal_ratio < 1.0:
        raise ValueError(f"val_normal_ratio deve estar em (0, 1), recebido {val_normal_ratio}.")
    if not 0.0 < val_attack_ratio < 1.0:
        raise ValueError(f"val_attack_ratio deve estar em (0, 1), recebido {val_attack_ratio}.")

    name = dataset_display_name(dataset_dir)
    rng = np.random.RandomState(random_state)

    path_x_train = os.path.join(dataset_dir, "X_train_baseline.parquet")
    path_y_train = os.path.join(dataset_dir, "y_train_baseline.parquet")
    path_x_test = os.path.join(dataset_dir, "X_test.parquet")
    path_y_test = os.path.join(dataset_dir, "y_test.parquet")
    path_x_ae = os.path.join(dataset_dir, "X_train_autoencoder.parquet")

    if verbose:
        print(f"[{name}] Lendo treino rotulado de '{os.path.basename(path_x_train)}'...")

    classes_train, detected_label_col = read_label_column(path_y_train, label_column)
    X_train, feature_columns = read_parquet_float32(path_x_train)

    if len(X_train) != len(classes_train):
        raise ValueError(
            f"[{name}] X_train_baseline ({len(X_train)} linhas) e y_train_baseline "
            f"({len(classes_train)} linhas) possuem tamanhos divergentes."
        )

    valid = _valid_label_mask(classes_train)
    if not valid.all():
        warnings.warn(
            f"[{name}] {int((~valid).sum())} linhas de treino com rótulo ausente foram descartadas.",
            RuntimeWarning,
            stacklevel=2,
        )
        X_train, classes_train = X_train[valid], classes_train[valid]

    resolved_normal = normal_label or detect_normal_label(classes_train)

    if max_train_rows is not None and max_train_rows < len(X_train):
        keep = _stratified_row_subsample(classes_train, int(max_train_rows), rng)
        X_train, classes_train = X_train[keep], classes_train[keep]
        if verbose:
            print(f"[{name}] Treino subamostrado de forma estratificada para {len(X_train):,} fluxos.")

    normal_mask = classes_train == resolved_normal
    idx_normal = np.flatnonzero(normal_mask)
    idx_attack = np.flatnonzero(~normal_mask)

    if len(idx_normal) < 10:
        raise ValueError(f"[{name}] Tráfego benigno insuficiente no treino: {len(idx_normal)} amostras.")

    # Consistência declarada do artefato: X_train_autoencoder deve ser exatamente
    # o subconjunto benigno de X_train_baseline.
    ae_file_check: Dict[str, Any] = {"present": os.path.exists(path_x_ae)}
    if ae_file_check["present"]:
        ae_rows = pq.ParquetFile(path_x_ae).metadata.num_rows
        ae_file_check["rows"] = int(ae_rows)
        ae_file_check["benign_rows_in_baseline"] = int(len(idx_normal))
        ae_file_check["row_count_matches"] = bool(max_train_rows is None and ae_rows == len(idx_normal))
        if max_train_rows is None and ae_rows != len(idx_normal):
            warnings.warn(
                f"[{name}] X_train_autoencoder possui {ae_rows} linhas, mas X_train_baseline "
                f"contém {len(idx_normal)} fluxos benignos. O treino do FC-DAE usará o "
                f"subconjunto benigno de X_train_baseline.",
                RuntimeWarning,
                stacklevel=2,
            )

    # --- Partição do tráfego benigno: treino (FC-DAE + DIF) vs validação normal ---
    shuffled_normal = rng.permutation(idx_normal)
    n_val_normal = int(round(val_normal_ratio * len(shuffled_normal)))
    n_val_normal = int(np.clip(n_val_normal, 1, len(shuffled_normal) - 1))
    idx_val_normal = shuffled_normal[:n_val_normal]
    idx_ae_train = shuffled_normal[n_val_normal:]

    # --- Partição estratificada dos ataques de treino: validação vs supervisionado ---
    idx_val_attacks: List[np.ndarray] = []
    idx_sup_attacks: List[np.ndarray] = []
    attack_split: Dict[str, Dict[str, int]] = {}

    for cls in np.unique(classes_train[idx_attack]):
        cls_idx = idx_attack[classes_train[idx_attack] == cls]
        count = len(cls_idx)
        shuffled = rng.permutation(cls_idx)

        n_val = int(round(val_attack_ratio * count))
        if count > 1:
            n_val = int(np.clip(n_val, 1, count - 1))
        else:
            n_val = count  # família singular: mantida na validação para ser observável

        idx_val_attacks.append(shuffled[:n_val])
        if n_val < count:
            idx_sup_attacks.append(shuffled[n_val:])
        attack_split[str(cls)] = {"total": count, "val": int(n_val), "sup_train": int(count - n_val)}

    idx_val_attack = np.concatenate(idx_val_attacks) if idx_val_attacks else np.empty(0, dtype=int)
    idx_sup_attack = np.concatenate(idx_sup_attacks) if idx_sup_attacks else np.empty(0, dtype=int)

    if len(idx_val_attack) == 0:
        raise ValueError(
            f"[{name}] Nenhum ataque disponível na validação. A calibração supervisionada "
            f"(Youden, alfa) requer ataques rotulados fora do conjunto de teste."
        )

    X_ae_train = X_train[idx_ae_train]
    X_ae_val = X_train[idx_val_normal]

    X_val = np.vstack([X_ae_val, X_train[idx_val_attack]])
    classes_val = np.concatenate([classes_train[idx_val_normal], classes_train[idx_val_attack]])
    y_val = (classes_val != resolved_normal).astype(int)
    val_normal_mask = y_val == 0

    X_sup_attacks = X_train[idx_sup_attack]
    classes_sup_attacks = classes_train[idx_sup_attack]

    del X_train  # libera a matriz completa de treino antes de carregar o teste

    # --- Conjunto de teste cego, lido na ordem de colunas do treino ---
    if verbose:
        print(f"[{name}] Lendo teste cego de '{os.path.basename(path_x_test)}'...")

    classes_test, test_label_col = read_label_column(path_y_test, label_column)
    X_test, _ = read_parquet_float32(path_x_test, columns=feature_columns)

    if test_label_col != detected_label_col:
        warnings.warn(
            f"[{name}] Coluna de rótulo divergente entre treino ('{detected_label_col}') "
            f"e teste ('{test_label_col}').",
            RuntimeWarning,
            stacklevel=2,
        )

    if len(X_test) != len(classes_test):
        raise ValueError(
            f"[{name}] X_test ({len(X_test)} linhas) e y_test ({len(classes_test)} linhas) divergem."
        )

    valid_test = _valid_label_mask(classes_test)
    if not valid_test.all():
        warnings.warn(
            f"[{name}] {int((~valid_test).sum())} linhas de teste com rótulo ausente foram descartadas.",
            RuntimeWarning,
            stacklevel=2,
        )
        X_test, classes_test = X_test[valid_test], classes_test[valid_test]

    if max_test_rows is not None and max_test_rows < len(X_test):
        keep = _stratified_row_subsample(classes_test, int(max_test_rows), rng)
        X_test, classes_test = X_test[keep], classes_test[keep]
        if verbose:
            print(f"[{name}] Teste subamostrado de forma estratificada para {len(X_test):,} fluxos.")

    y_test = (classes_test != resolved_normal).astype(int)

    rare_classes = {
        str(cls): int(np.sum(classes_test == cls))
        for cls in np.unique(classes_test[y_test == 1])
        if int(np.sum(classes_test == cls)) < rare_threshold
    }

    meta: Dict[str, Any] = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "label_column": detected_label_col,
        "normal_label": resolved_normal,
        "n_features": len(feature_columns),
        "protocol": {
            "val_normal_ratio": val_normal_ratio,
            "val_attack_ratio": val_attack_ratio,
            "rare_threshold": rare_threshold,
            "random_state": random_state,
            "max_train_rows": max_train_rows,
            "max_test_rows": max_test_rows,
        },
        "attack_split_train": attack_split,
        "train_class_counts": {str(c): int(n) for c, n in zip(*np.unique(classes_train, return_counts=True))},
        "test_class_counts": {str(c): int(n) for c, n in zip(*np.unique(classes_test, return_counts=True))},
        "autoencoder_file_check": ae_file_check,
        "preprocessor_present": os.path.exists(os.path.join(dataset_dir, "preprocessor.joblib")),
        # Procedência do split fornecido: distingue particionamento canônico de
        # refatiamento estratificado (relevante para o cenário zero-day).
        "split_provenance": diagnose_split_provenance(classes_train, classes_test),
        # Viabilidade do baseline B5: exige carimbo temporal para o janelamento.
        "temporal_support": {
            "temporal_columns_in_features": detect_temporal_columns(feature_columns),
            "sidecar_files_present": sorted(
                entry
                for entry in os.listdir(dataset_dir)
                if any(k in entry.lower() for k in ("timestamp", "metadado", "metadata", "time"))
            ),
            "b5_lstm_executable": False,
        },
        # Proporções globais (treino + teste) de cada partição, para que o texto do TCC
        # reporte o particionamento efetivo em vez do esquema nominal 70/15/15.
        "global_partition_share": {},
    }

    temporal = meta["temporal_support"]
    temporal["b5_lstm_executable"] = bool(
        temporal["temporal_columns_in_features"] or temporal["sidecar_files_present"]
    )

    partitions = NidsPartitions(
        dataset_name=name,
        label_column=detected_label_col,
        normal_label=resolved_normal,
        feature_columns=feature_columns,
        X_ae_train=X_ae_train,
        X_ae_val=X_ae_val,
        X_val=X_val,
        y_val=y_val,
        classes_val=classes_val,
        val_normal_mask=val_normal_mask,
        X_sup_attacks=X_sup_attacks,
        classes_sup_attacks=classes_sup_attacks,
        X_test=X_test,
        y_test=y_test,
        classes_test=classes_test,
        rare_classes=rare_classes,
        meta=meta,
    )
    meta["sizes"] = partitions.sizes()

    # Proporções efetivas sobre o total (treino + teste) de cada classe binária. O
    # esquema nominal 70/15/15 do protocolo pressupõe liberdade para reparticionar
    # tudo; aqui o split treino/teste vem fixado pelo artefato, então as frações reais
    # são estas — e é este o número que o texto deve reportar.
    sizes = meta["sizes"]
    total_normal = sizes["ae_train_normal"] + sizes["ae_val_normal"] + sizes["test_normal"]
    total_attack = sizes["val_attacks"] + sizes["sup_train_attacks"] + sizes["test_attacks"]
    meta["global_partition_share"] = {
        "normal": {
            "train_unsupervised": sizes["ae_train_normal"] / max(1, total_normal),
            "validation": sizes["ae_val_normal"] / max(1, total_normal),
            "test": sizes["test_normal"] / max(1, total_normal),
            "total": total_normal,
        },
        "attack": {
            "validation": sizes["val_attacks"] / max(1, total_attack),
            "reserved_supervised": sizes["sup_train_attacks"] / max(1, total_attack),
            "test": sizes["test_attacks"] / max(1, total_attack),
            "total": total_attack,
        },
    }

    if verbose:
        sizes = meta["sizes"]
        print(
            f"[{name}] n={len(feature_columns)} features | rótulo='{detected_label_col}' "
            f"| benigno='{resolved_normal}'"
        )
        print(
            f"[{name}] FC-DAE/DIF treino benigno: {sizes['ae_train_normal']:,} | "
            f"validação benigna: {sizes['ae_val_normal']:,} | "
            f"validação rotulada: {sizes['val_total']:,} "
            f"({sizes['val_attacks']:,} ataques) | "
            f"ataques p/ B4: {sizes['sup_train_attacks']:,} | "
            f"teste cego: {sizes['test_total']:,} ({sizes['test_attacks']:,} ataques)"
        )
        if rare_classes:
            print(f"[{name}] Famílias raras no teste (<{rare_threshold}): {rare_classes}")

    return partitions

"""Extração do espaço latente (z) e do erro de reconstrução MSE por amostra.

Corresponde ao Módulo 3 (Pre-IF): o *bottleneck* do FC-DAE treinado é isolado
como extrator determinístico, produzindo simultaneamente o vetor latente
`z ∈ R^m` e o MSE de reconstrução de cada fluxo. A inferência é feita em
mini-batches para manter o pico de memória constante em datasets grandes.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.autoencoder import FCDAE


@torch.no_grad()
def extract_latent_and_mse(
    model: FCDAE,
    X: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Projeta `X` no espaço latente e calcula o MSE de reconstrução por amostra.

    :returns: par `(Z, mse)` com `Z` em `float32` de forma `(n, m)` e `mse` em
        `float64` de forma `(n,)`.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X deve ser uma matriz 2D; recebido {X.shape}.")

    device = device or next(model.parameters()).device
    model.eval()

    n_samples = X.shape[0]
    Z = np.empty((n_samples, int(model.bottleneck_dim)), dtype=np.float32)
    mse = np.empty(n_samples, dtype=np.float64)

    step = max(1, int(batch_size))
    for start in range(0, n_samples, step):
        end = min(start + step, n_samples)
        batch = torch.from_numpy(X[start:end]).to(device, non_blocking=True)
        reconstruction, latent = model(batch)
        batch_mse = torch.mean((reconstruction - batch) ** 2, dim=1)

        Z[start:end] = latent.detach().cpu().numpy()
        mse[start:end] = batch_mse.detach().cpu().numpy().astype(np.float64)

    return Z, mse


def latent_frame(
    Z: np.ndarray,
    mse: np.ndarray,
    classes: Optional[Sequence[object]] = None,
    label_column: str = "Label",
) -> pd.DataFrame:
    """Monta o DataFrame do *espaço latente persistido* (`z_0..z_{m-1}`, `mse`, rótulo)."""
    frame = pd.DataFrame(Z, columns=[f"z_{i}" for i in range(Z.shape[1])])
    frame["mse"] = np.asarray(mse, dtype=np.float64)
    if classes is not None:
        frame[label_column] = np.asarray(classes, dtype=object)
    return frame


def save_latent_parquet(
    path: str,
    Z: np.ndarray,
    mse: np.ndarray,
    classes: Optional[Sequence[object]] = None,
    label_column: str = "Label",
) -> str:
    """Persiste o espaço latente em `.parquet` para consumo do DIF e do t-SNE."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    latent_frame(Z, mse, classes=classes, label_column=label_column).to_parquet(path, index=False)
    return os.path.abspath(path)

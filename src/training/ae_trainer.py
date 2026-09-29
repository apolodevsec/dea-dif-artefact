"""Rotina de treino e validação do FC-DAE (Módulo 2) sobre tráfego benigno.

O treino é estritamente não-supervisionado: o modelo minimiza o erro de
reconstrução do tráfego legítimo e é monitorado por uma partição benigna de
validação disjunta, que governa o *early stopping*, o `ReduceLROnPlateau` e a
seleção do melhor checkpoint. Nenhum ataque participa desta etapa.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import FCDAE
from src.training.device import resolve_device


@dataclass
class AETrainConfig:
    """Hiperparâmetros da rotina de treino do FC-DAE."""

    epochs: int = 50
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    min_delta: float = 0.0
    lr_factor: float = 0.5
    lr_patience: int = 5
    bottleneck_dim: Optional[int] = None
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42
    grad_clip_norm: Optional[float] = None
    verbose: bool = True


@dataclass
class AETrainResult:
    """Histórico e artefatos produzidos pela rotina de treino."""

    model: FCDAE
    device: torch.device
    input_dim: int
    bottleneck_dim: int
    best_val_loss: float
    best_epoch: int
    epochs_run: int
    early_stopped: bool
    train_seconds: float
    history: List[Dict[str, float]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_report(self) -> Dict[str, Any]:
        """Serializa o resultado para JSON (sem o objeto do modelo)."""
        return {
            "input_dim": self.input_dim,
            "bottleneck_dim": self.bottleneck_dim,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_run": self.epochs_run,
            "early_stopped": self.early_stopped,
            "train_seconds": self.train_seconds,
            "device": str(self.device),
            "checkpoint_path": self.checkpoint_path,
            "config": self.config,
            "history": self.history,
        }


def seed_everything(seed: int) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_bottleneck_dim(n_features: int) -> int:
    """Dimensão do *bottleneck* adotada pelo framework: m = floor(n/8) (ou n/4 se n < 50)."""
    if n_features < 50:
        return max(1, n_features // 4)
    return max(1, n_features // 8)


def _evaluate_loss(
    model: FCDAE,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Erro de reconstrução médio por amostra sobre um loader completo."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            total_loss += float(loss.item()) * batch.size(0)
            total_samples += int(batch.size(0))
    return total_loss / max(1, total_samples)


def train_fc_dae(
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: Optional[AETrainConfig] = None,
    checkpoint_path: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    extra_checkpoint_fields: Optional[Dict[str, Any]] = None,
) -> AETrainResult:
    """Treina o FC-DAE em `X_train` monitorando `X_val` (ambos estritamente benignos).

    :param X_train: matriz `(n_train, n_features)` de tráfego benigno normalizado.
    :param X_val: matriz `(n_val, n_features)` benigna e disjunta de `X_train`.
    :param checkpoint_path: se informado, persiste o melhor estado do modelo.
    :returns: `AETrainResult` com o modelo em melhor época e o histórico completo.
    """
    config = config or AETrainConfig()

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_val = np.ascontiguousarray(X_val, dtype=np.float32)

    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError(
            f"X_train e X_val devem ser matrizes 2D; recebidos {X_train.shape} e {X_val.shape}."
        )
    if len(X_train) == 0:
        raise ValueError("X_train está vazio: não há tráfego benigno para treinar o FC-DAE.")
    if len(X_val) == 0:
        raise ValueError(
            "X_val está vazio: a validação benigna é obrigatória para o early stopping "
            "e para a calibração de tau_AE."
        )
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Número de features divergente entre treino ({X_train.shape[1]}) e validação ({X_val.shape[1]})."
        )

    seed_everything(config.seed)
    device = resolve_device(config.device, verbose=config.verbose)

    n_features = int(X_train.shape[1])
    bottleneck = config.bottleneck_dim or default_bottleneck_dim(n_features)
    if bottleneck < 1 or bottleneck > n_features:
        raise ValueError(f"bottleneck_dim inválido: {bottleneck} para n={n_features} features.")

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train)),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = FCDAE(input_dim=n_features, bottleneck_dim=bottleneck).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
    )

    if config.verbose:
        print(
            f"FC-DAE instanciado: n={n_features} -> {n_features // 2} -> {n_features // 4} "
            f"-> m={model.bottleneck_dim} (bottleneck linear) -> ... -> n (sigmoide)"
        )
        print(
            f"Treino benigno: {len(X_train):,} fluxos | validação benigna: {len(X_val):,} fluxos "
            f"| batch={config.batch_size} | lr={config.lr} | paciência={config.patience}"
        )

    best_val_loss = math.inf
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    early_stopped = False
    history: List[Dict[str, float]] = []

    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for (batch,) in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            running_loss += float(loss.item()) * batch.size(0)
            seen += int(batch.size(0))

        train_loss = running_loss / max(1, seen)
        val_loss = _evaluate_loss(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        improved = val_loss < (best_val_loss - config.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config.verbose:
            marker = " *" if improved else ""
            print(
                f"Época {epoch:03d}/{config.epochs:03d} | treino: {train_loss:.6f} "
                f"| validação: {val_loss:.6f} | lr: {current_lr:.2e}{marker}"
            )

        if epochs_no_improve >= config.patience:
            early_stopped = True
            if config.verbose:
                print(
                    f"Early stopping na época {epoch} ({config.patience} épocas sem melhora). "
                    f"Melhor época: {best_epoch}."
                )
            break

    train_seconds = time.perf_counter() - started

    # Restaura os pesos da melhor época para que a extração latente use o modelo ótimo.
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    result = AETrainResult(
        model=model,
        device=device,
        input_dim=n_features,
        bottleneck_dim=int(model.bottleneck_dim),
        best_val_loss=float(best_val_loss),
        best_epoch=int(best_epoch),
        epochs_run=len(history),
        early_stopped=early_stopped,
        train_seconds=float(train_seconds),
        history=history,
        config=asdict(config),
    )

    if checkpoint_path:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "input_dim": n_features,
            "bottleneck_dim": int(model.bottleneck_dim),
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "feature_columns": list(feature_columns) if feature_columns is not None else None,
            "config": asdict(config),
        }
        if extra_checkpoint_fields:
            payload.update(extra_checkpoint_fields)
        torch.save(payload, checkpoint_path)
        result.checkpoint_path = os.path.abspath(checkpoint_path)
        if config.verbose:
            print(f"Checkpoint da melhor época salvo em '{checkpoint_path}' (val={best_val_loss:.6f}).")

    return result


def save_training_history(result: AETrainResult, path: str) -> str:
    """Persiste o histórico de treino em JSON para os gráficos de convergência."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_report(), handle, indent=2, ensure_ascii=False)
    return os.path.abspath(path)

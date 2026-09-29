"""Resolução do dispositivo de computação (CUDA / DirectML / CPU multi-thread)."""

from __future__ import annotations

import os

import torch


def resolve_device(requested: str = "auto", verbose: bool = True) -> torch.device:
    """Resolve o dispositivo de execução do PyTorch.

    :param requested: `auto`, `cpu`, `cuda`, `cuda:N` ou `dml` (DirectML).
    """
    requested = (requested or "auto").strip().lower()

    if requested == "dml":
        try:
            import torch_directml

            if verbose:
                print("Dispositivo: GPU via DirectML (AMD/Intel/NVIDIA).")
            return torch_directml.device()
        except ImportError:
            if verbose:
                print("Aviso: torch-directml indisponível. Recuando para CPU.")
            return _cpu_device(verbose)

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        if verbose:
            print(f"Dispositivo: GPU CUDA ({torch.cuda.get_device_name(0)}).")
        return torch.device("cuda")

    try:
        import torch_directml

        if verbose:
            print("Dispositivo: GPU via DirectML.")
        return torch_directml.device()
    except ImportError:
        pass

    return _cpu_device(verbose)


def _cpu_device(verbose: bool) -> torch.device:
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    if verbose:
        print(f"Dispositivo: CPU com {num_threads} threads.")
    return torch.device("cpu")

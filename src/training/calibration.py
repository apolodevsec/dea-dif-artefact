"""Calibração de limiares do framework híbrido.

Dois regimes coexistem no protocolo e não devem ser confundidos:

- **Não-supervisionado** (`tau_AE`, `tau_DIF_ref`): derivado apenas de estatísticas
  do tráfego benigno de validação (percentil `p` ou `mu + k*sigma`). É o limiar
  operacional realista, pois não exige ataques rotulados.
- **Supervisionado** (`tau_opt`, alfa da fusão): otimizado pelo Índice de Youden
  sobre a validação rotulada. Serve de teto comparativo e é calibrado fora do
  conjunto de teste cego.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

# Formulações aceitas para o limiar do autoencoder. Ver `calibrate_ae_thresholds`.
AE_THRESHOLD_RULES = ("percentile", "mu_sigma")


def mu_sigma_threshold(values: Sequence[float], k: float = 3.0) -> float:
    """Limiar estatístico `mu + k*sigma` sobre o erro do tráfego benigno de validação."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Vetor vazio: impossível calibrar o limiar mu + k*sigma.")
    return float(arr.mean() + k * arr.std(ddof=0))


def percentile_threshold(values: Sequence[float], percentile: float = 95.0) -> float:
    """Limiar por percentil sobre o erro/escore do tráfego benigno de validação."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Vetor vazio: impossível calibrar o limiar por percentil.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentil deve estar em [0, 100]; recebido {percentile}.")
    return float(np.percentile(arr, percentile))


def calibrate_ae_thresholds(
    mse_val_normal: Sequence[float],
    k_sigma: float = 3.0,
    percentile: float = 95.0,
    rule: str = "percentile",
) -> Dict[str, Any]:
    """Calibra `tau_AE` sobre o MSE do tráfego benigno de validação.

    As duas formulações usadas no framework **não são equivalentes**: `mu + k*sigma`
    só coincide com um percentil fixo se o erro de reconstrução for aproximadamente
    normal, o que tipicamente não ocorre — a distribuição do MSE é assimétrica à
    direita e de cauda longa. Por isso ambas são sempre calculadas e reportadas, mas
    `tau_ae_ref` (o limiar não-supervisionado efetivamente adotado) segue `rule`:

    - `"percentile"` (padrão): `tau_AE` = `p`-ésimo percentil, com `p = 95` por
      omissão. É a formulação da Eq. 4 / Seção 4.7.1 do TCC e a que a varredura OFAT
      do fator `p` percorre (`p` em {90, 95, 97, 99}).
    - `"mu_sigma"`: `tau_AE = mu + k*sigma`, formulação alternativa registrada em
      `ae.md`. Ao adotá-la, a análise de sensibilidade deve varrer `k`, não `p`.

    :raises ValueError: se `rule` não estiver em `AE_THRESHOLD_RULES`.
    """
    if rule not in AE_THRESHOLD_RULES:
        raise ValueError(f"rule deve ser um de {AE_THRESHOLD_RULES}; recebido '{rule}'.")

    arr = np.asarray(mse_val_normal, dtype=np.float64)
    tau_sigma = mu_sigma_threshold(arr, k=k_sigma)
    tau_pct = percentile_threshold(arr, percentile=percentile)

    return {
        "mse_val_normal_mean": float(arr.mean()),
        "mse_val_normal_std": float(arr.std(ddof=0)),
        "k_sigma": float(k_sigma),
        "percentile": float(percentile),
        "tau_ae_mu_sigma": tau_sigma,
        "tau_ae_percentile": tau_pct,
        "tau_ae_rule": rule,
        "tau_ae_ref": tau_pct if rule == "percentile" else tau_sigma,
        # Fração do benigno de validação rejeitada por cada critério: explicita o
        # quanto os dois divergem no FPR esperado, já que não são intercambiáveis.
        "expected_fpr_mu_sigma": float(np.mean(arr > tau_sigma)),
        "expected_fpr_percentile": float(np.mean(arr > tau_pct)),
        # Percentil empírico em que mu + k*sigma efetivamente cai. Longe de 99 indica
        # violação material da suposição de normalidade do MSE.
        "mu_sigma_empirical_percentile": float(np.mean(arr <= tau_sigma) * 100.0),
    }

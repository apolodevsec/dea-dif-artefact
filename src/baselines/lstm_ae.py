import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, Tuple

from src.evaluation.hybrid import _find_best_youden_threshold


class _LSTMAEModule(nn.Module):
    """Módulo neural LSTM Autoencoder com codificação e decodificação sequencial."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.enc_linear = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM
        self.dec_linear = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape

        _, (h_n, _) = self.encoder_lstm(x)
        # h_n[-1]: (batch, hidden_dim)
        latent = torch.relu(self.enc_linear(h_n[-1]))  # (batch, latent_dim)

        dec_init = torch.relu(self.dec_linear(latent))  # (batch, hidden_dim)
        # Repete o vetor latente para todos os passos da sequência
        rep_latent = dec_init.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        dec_out, _ = self.decoder_lstm(rep_latent)
        reconstruction = self.output_linear(dec_out)  # (batch, seq_len, input_dim)

        return reconstruction


class LSTMAutoencoderDetector:
    """Baseline B5: LSTM Autoencoder para detecção de anomalias em janelas sequenciais.
    
    Nota Metodológica sobre Assimetria de Lookback:
    Detectores baseados em janelas temporais exploram contexto temporal de W-1 fluxos precedentes,
    diferindo conceitualmente dos detectores tabulares baseados em conexões isoladas.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 8,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.005,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        torch.manual_seed(random_state)
        self.model = _LSTMAEModule(input_dim, hidden_dim, latent_dim).to(self.device)
        self.tau_opt_: float = 0.01
        self.best_youden_: float = 0.0

    def fit(self, X_seq: np.ndarray) -> "LSTMAutoencoderDetector":
        """Treina o autoencoder recorrente exclusivamente com sequências normais."""
        self.model.train()
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()

        self.model.eval()
        return self

    def score_samples(self, X_seq: np.ndarray, trailing_only: bool = False) -> np.ndarray:
        """Calcula o erro de reconstrução MSE por sequência."""
        self.model.eval()
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon = self.model(X_t)
            if trailing_only:
                # Erro apenas no último fluxo da janela (trailing connection)
                diff = recon[:, -1, :] - X_t[:, -1, :]
                mse = torch.mean(diff ** 2, dim=-1).cpu().numpy()
            else:
                # Erro médio em toda a janela temporal
                diff = recon - X_t
                mse = torch.mean(diff ** 2, dim=(1, 2)).cpu().numpy()

        return mse.astype(np.float64)

    def calibrate(self, X_seq_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Calibra o limiar ótimo de detecção sobre a validação rotulada."""
        scores_val = self.score_samples(X_seq_val)
        tau, j = _find_best_youden_threshold(np.asarray(y_val, dtype=int), scores_val)
        self.tau_opt_ = float(tau)
        self.best_youden_ = float(j)
        return {"tau_opt": self.tau_opt_, "best_youden": self.best_youden_}

    def predict(self, X_seq: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Classifica as sequências como anômalas (1) ou normais (0)."""
        scores = self.score_samples(X_seq)
        tau = self.tau_opt_ if threshold is None else threshold
        return (scores > tau).astype(int)
